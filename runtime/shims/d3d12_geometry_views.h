#pragma once

#include "d3d12_title_buffer_windows.h"
#include "../src/title_geometry_watch.h"
#include <chrono>
#include <map>
#include <mutex>
#include <set>

namespace rt::d3d12 {

// One persistent device-local copy for each geometry extent. The console heap retains its original indices.
// Every CPU mapping of a source tile is watched; a physical remap changes which aliases must be watched.
enum class GeometryCopy { copied, unreadable, interrupted };

class GeometryViews {
    using Address = std::uintptr_t;
    static constexpr auto tile_size = leonida::geometry_writes::tile_size;
    using Cell = leonida::geometry_writes::Cell;
    struct View {
        Address source;
        std::size_t size;
        std::uint8_t* dst;
        VkDeviceSize va;
        bool ready{}, zeroed{};
    };
    struct Tile {
        Cell* primary{};
        std::uint64_t mapping{}, version{};
        std::vector<std::pair<Address, Cell*>> peers;
        std::vector<std::size_t> views;
        bool retry = false;
    };
    mutable std::mutex mutex_;
    TitleBufferWindows pool_;
    std::map<std::pair<Address, std::size_t>, std::size_t> keys_;
    std::vector<View> views_;
    std::map<VkDeviceAddress, std::size_t> by_address_;
    std::map<Address, Tile> tiles_;
    static constexpr auto refused = SIZE_MAX;
    std::uint64_t serial_{}, bytes_{}, cap_refusals_{}, allocation_failures_{}, ready_{}, uploads_{}, uploaded_{};
    std::uint64_t submissions_{}, dirty_refreshes_{}, refilled_{};
    std::chrono::steady_clock::duration refresh_time_{};
    std::uint64_t cleared_{};
    bool retry_{};

    bool update(Address va, Tile& t) {
        namespace watch = leonida::geometry_writes;
        if (!t.primary) t.primary = watch::acquire(va);
        auto version = t.primary->version.load(std::memory_order_acquire);
        const auto mapping = t.primary->mapping.load(std::memory_order_acquire);
        if (t.mapping != mapping) {
            std::vector<Address> others;
            if (auto aliases = watch::aliases.load(std::memory_order_acquire)) aliases(va, others);
            auto old = std::move(t.peers);
            for (auto peer : others) t.peers.emplace_back(peer, watch::acquire(peer));
            for (auto [peer, cell] : old) watch::release(peer);
            t.mapping = mapping;
        }
        for (auto [addr, peer] : t.peers)
            version = (std::max)(version, peer->version.load(std::memory_order_acquire));
        if (version == t.version && !t.retry) return false;
        t.retry = !watch::arm(va, *t.primary);
        for (auto [addr, peer] : t.peers)
            if (!watch::arm(addr, *peer)) t.retry = true;
        t.version = version;
        retry_ |= t.retry;
        return !t.retry;
    }
    template<class Copy>
    void fill(View& v, Copy& copy) {
        const auto result = copy(v.dst, reinterpret_cast<const std::uint8_t*>(v.source), v.size);
        if (result == GeometryCopy::copied) {
            if (!v.ready) ++ready_;
            v.ready = true;
            v.zeroed = false;
            ++uploads_;
            uploaded_ += v.size;
        } else {
            if (v.ready) --ready_;
            v.ready = false;
            if (!v.zeroed || result == GeometryCopy::interrupted) {
                // A failing guarded copy may have written a prefix. A failed readability probe
                // wrote nothing, so an already-zeroed view needs no second write across the BAR.
                std::memset(v.dst, 0, v.size);
                cleared_ += v.size;
                v.zeroed = true;
            }
        }
    }

public:
    static std::uint64_t cap() {
        static const std::uint64_t value = []() -> std::uint64_t {
            const char* opt = std::getenv("LEONIDA_GEOMETRY_MB");
            return opt ? (std::min)(std::strtoull(opt, nullptr, 10), 8192ull) << 20 : 0ull;
        }();
        return value;
    }

    std::uint64_t find(Address cpu, std::size_t bytes) const {
        std::lock_guard lock(mutex_);
        const auto it = keys_.find({cpu, bytes});
        return it == keys_.end() || it->second == refused ? 0 : views_[it->second].va;
    }

    template<class Device, class Copy>
    std::uint64_t resolve(Device& device, Address cpu, std::size_t bytes, Copy copy) {
        if (!bytes || bytes > UINTPTR_MAX - cpu) return 0;
        std::lock_guard lock(mutex_);
        const auto key = std::make_pair(cpu, bytes);
        if (const auto it = keys_.find(key); it != keys_.end())
            return it->second == refused ? 0 : views_[it->second].va;
        std::uint8_t* mapped{};
        VkDeviceSize va{};
        bool at_cap = false;
        if (bytes > cap() - bytes_ || !pool_.allocate(device, bytes, va, mapped, true, cap(), &at_cap)) {
            if (at_cap || bytes > cap() - bytes_) {
                keys_.emplace(key, refused);
                if (++cap_refusals_ <= 4)
                    std::fprintf(stderr, "[rt] geometry buffer cap refused %zu bytes, %llu bytes already reserved\n",
                                 bytes, static_cast<unsigned long long>(pool_.reserved()));
            } else {
                ++allocation_failures_;
            }
            return 0;
        }
        auto id = views_.size();
        keys_[key] = id;
        by_address_.emplace(device.arena.address + va, id);
        views_.push_back({cpu, bytes, mapped, va});
        bytes_ += bytes;
        bool ready = true;
        for (auto p = cpu & ~(tile_size - 1); p < cpu + bytes; p += tile_size) {
            auto& tile = tiles_[p];
            // Previously watched tiles may have other dirty views waiting for refresh().
            if (!tile.primary) update(p, tile);
            tile.views.push_back(id);
            ready &= !tile.retry;
        }
        if (ready) fill(views_.back(), copy);
        else {
            std::memset(mapped, 0, bytes);
            cleared_ += bytes;
            views_.back().zeroed = true;
        }
        return va;
    }

    template<class Copy>
    void refresh(Copy copy) {
        if (!cap()) return;
        std::lock_guard lock(mutex_);
        ++submissions_;
        auto now = leonida::geometry_writes::state().serial.load(std::memory_order_acquire);
        if (serial_ == now && !retry_) return;
        const auto start = std::chrono::steady_clock::now();
        const auto before = uploaded_;
        serial_ = now;
        retry_ = false;
        std::set<std::size_t> changed;
        for (auto& [va, tile] : tiles_)
            if (update(va, tile)) changed.insert(tile.views.begin(), tile.views.end());
        for (auto id : changed) {
            auto& view = views_[id];
            bool pinned = false;
            for (auto p = view.source & ~(tile_size - 1); p < view.source + view.size; p += tile_size)
                if (tiles_.at(p).retry) { pinned = true; break; }
            if (!pinned) fill(view, copy);
        }
        if (uploaded_ != before) ++dirty_refreshes_;
        refilled_ += uploaded_ - before;
        refresh_time_ += std::chrono::steady_clock::now() - start;
    }

    bool lookup(VkDeviceAddress bda, std::uint64_t& host, std::uint64_t& bytes, std::uint64_t* source = nullptr,
                bool* populated = nullptr) const {
        std::lock_guard lock(mutex_);
        auto it = by_address_.upper_bound(bda);
        if (it == by_address_.begin()) return false;
        --it;
        const auto offset = bda - it->first;
        const auto& v = views_[it->second];
        if (offset >= v.size) return false;
        bytes = v.size - offset;
        host = reinterpret_cast<Address>(v.dst) + offset;
        if (source) *source = v.source + offset;
        if (populated) *populated = v.ready;
        return true;
    }
    void report() const {
        if (!cap()) return;
        std::lock_guard lock(mutex_);
        std::fprintf(stderr, "[rt] geometry buffers: seen %zu, backed %zu (%llu bytes), populated %llu, cap %llu MiB, "
                             "cap refused %llu, allocation failures %llu, uploads %llu (%llu bytes), watched tiles %zu, reserved %llu bytes\n",
                     keys_.size(), views_.size(), (unsigned long long)bytes_, (unsigned long long)ready_,
                     (unsigned long long)(cap() >> 20), (unsigned long long)cap_refusals_, (unsigned long long)allocation_failures_, (unsigned long long)uploads_,
                     (unsigned long long)uploaded_, tiles_.size(),
                     static_cast<unsigned long long>(pool_.reserved()));
        std::fprintf(stderr, "[rt] geometry refresh: %llu/%llu submissions uploaded dirty views, %llu bytes; "
                             "%.3f s checking and refreshing sources, %llu bytes cleared\n",
                     static_cast<unsigned long long>(dirty_refreshes_), static_cast<unsigned long long>(submissions_),
                     static_cast<unsigned long long>(refilled_), std::chrono::duration<double>(refresh_time_).count(),
                     static_cast<unsigned long long>(cleared_));
    }
    template<class Device>
    void release(Device& d) {
        for (auto& [va, tile] : tiles_) {
            for (auto [peer, cell] : tile.peers) leonida::geometry_writes::release(peer);
            leonida::geometry_writes::release(va);
        }
        tiles_.clear();
        pool_.release(d);
    }
};
} // namespace rt::d3d12
