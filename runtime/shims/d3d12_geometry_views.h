#pragma once

#include "d3d12_title_buffer_windows.h"
#include "../src/title_geometry_watch.h"
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <cstring>
#include <unordered_map>

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
    std::mutex refresh_mutex_;  // refresh() only, held for its whole run, before mutex_
    TitleBufferWindows pool_;
    std::map<std::pair<Address, std::size_t>, std::size_t> keys_;
    // What resolve() and find() answer for a key already in keys_ (0 when refused), readable without mutex_: refresh()
    // holds mutex_ for its whole pass over the watched tiles (up to tens of milliseconds a submission), and the title's
    // Subrender workers resolve their root views while they record (SetGraphicsRoot*View), so they waited for it
    // (speed-26). Written under mutex_ and then fast_mutex_ (in that order), read under fast_mutex_ alone.
    struct KeyHash {
        std::size_t operator()(const std::pair<Address, std::size_t>& k) const {
            return std::hash<std::uint64_t>{}(std::uint64_t(k.first) * 0x9E3779B97F4A7C15ull ^ k.second);
        }
    };
    // keys_, views_ (the vector itself, and each view's source, size, dst and va) and by_address_ change only in
    // resolve(), under mutex_ and then structure_mutex_ exclusively. lookup() and find() read them under
    // structure_mutex_ shared, so they do not wait for a refresh() in progress (which holds mutex_ and changes only a
    // view's ready/zeroed flags and its bytes), on the submit thread's heaps refresh among others.
    mutable std::shared_mutex structure_mutex_;
    mutable std::shared_mutex fast_mutex_;
    std::unordered_map<std::pair<Address, std::size_t>, VkDeviceSize, KeyHash> fast_;
    void publish(const std::pair<Address, std::size_t>& key, VkDeviceSize va) {  // mutex_ held
        std::unique_lock lock(fast_mutex_);
        fast_[key] = va;
    }
    bool fast_find(const std::pair<Address, std::size_t>& key, VkDeviceSize& va) const {
        std::shared_lock lock(fast_mutex_);
        const auto it = fast_.find(key);
        if (it == fast_.end()) return false;
        va = it->second;
        return true;
    }
    std::vector<View> views_;
    std::map<VkDeviceAddress, std::size_t> by_address_;
    std::map<Address, Tile> tiles_;
    static constexpr auto refused = SIZE_MAX;
    std::uint64_t serial_{}, bytes_{}, cap_refusals_{}, allocation_failures_{}, ready_{}, uploads_{}, uploaded_{};
    std::uint64_t submissions_{}, dirty_refreshes_{}, refilled_{}, tile_fills_{};
    std::unordered_map<Address, std::size_t> tile_reads_;  // refresh(): offset in tile_scratch_, SIZE_MAX unreadable
    // refresh() scratch: per view, whether it waits on a tile (pinned), is filled whole, was touched, was counted;
    // and the tiles found written, in address order.
    enum : std::uint8_t { kPinned = 1, kWhole = 2, kTouched = 4, kCounted = 8 };
    std::vector<std::uint8_t> marks_;
    std::vector<const Tile*> written_;
    std::vector<Address> written_vas_;
    std::vector<std::uint8_t> tile_scratch_;
    std::chrono::steady_clock::duration refresh_time_{};
    struct Window {  // since the last periodic line
        std::uint64_t refreshes = 0, tiles = 0, views = 0, bytes = 0;
        std::chrono::steady_clock::duration time{};
        std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    } window_;
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
    // The tile's 64 KiB as this refresh read it, or null when it could not be read whole. Read at most once a refresh.
    template<class Copy>
    const std::uint8_t* read_tile(Address tile, Copy& copy) {
        auto [it, fresh] = tile_reads_.try_emplace(tile, SIZE_MAX);
        if (fresh) {
            const std::size_t at = tile_scratch_.size();
            tile_scratch_.resize(at + tile_size);
            if (copy(tile_scratch_.data() + at, reinterpret_cast<const std::uint8_t*>(tile), tile_size) ==
                GeometryCopy::copied)
                it->second = at;
            else
                tile_scratch_.resize(at);
        }
        return it->second == SIZE_MAX ? nullptr : tile_scratch_.data() + it->second;
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
        if (VkDeviceSize va = 0; fast_find({cpu, bytes}, va)) return va;
        std::shared_lock lock(structure_mutex_);
        const auto it = keys_.find({cpu, bytes});
        return it == keys_.end() || it->second == refused ? 0 : views_[it->second].va;
    }

    template<class Device, class Copy>
    std::uint64_t resolve(Device& device, Address cpu, std::size_t bytes, Copy copy) {
        if (!bytes || bytes > UINTPTR_MAX - cpu) return 0;
        const auto key = std::make_pair(cpu, bytes);
        if (VkDeviceSize known = 0; fast_find(key, known)) return known;
        std::lock_guard lock(mutex_);
        if (const auto it = keys_.find(key); it != keys_.end()) {
            const VkDeviceSize known = it->second == refused ? 0 : views_[it->second].va;
            publish(key, known);
            return known;
        }
        std::uint8_t* mapped{};
        VkDeviceSize va{};
        bool at_cap = false;
        if (bytes > cap() - bytes_ || !pool_.allocate(device, bytes, va, mapped, true, cap(), &at_cap)) {
            if (at_cap || bytes > cap() - bytes_) {
                {
                    std::unique_lock structure(structure_mutex_);
                    keys_.emplace(key, refused);
                }
                publish(key, 0);
                if (++cap_refusals_ <= 4)
                    std::fprintf(stderr, "[rt] geometry buffer cap refused %zu bytes, %llu bytes already reserved\n",
                                 bytes, static_cast<unsigned long long>(pool_.reserved()));
            } else {
                ++allocation_failures_;
            }
            return 0;
        }
        auto id = views_.size();
        {
            std::unique_lock structure(structure_mutex_);
            keys_[key] = id;
            by_address_.emplace(device.arena.address + va, id);
            views_.push_back({cpu, bytes, mapped, va});
        }
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
        publish(key, va);
        return va;
    }

    template<class Copy>
    void refresh(Copy copy) {
        if (!cap()) return;
        // One refresh at a time (its scratch members, and mutex_ is let go during the walk below); taken before mutex_.
        std::lock_guard serial(refresh_mutex_);
        std::unique_lock lock(mutex_);
        ++submissions_;
        auto now = leonida::geometry_writes::state().serial.load(std::memory_order_acquire);
        if (serial_ == now && !retry_) return;
        const auto start = std::chrono::steady_clock::now();
        const auto before = uploaded_;
        serial_ = now;
        retry_ = false;
        // The views over each tile written since, with those tiles (in address order, as tiles_ is): a ready view
        // copies only them, 64 KiB each, not the whole view again (LEONIDA_GEOMETRY_WHOLE_VIEWS=1: whole views).
        static const bool whole = [] {
            const char* v = std::getenv("LEONIDA_GEOMETRY_WHOLE_VIEWS");
            return v && *v && *v != '0';
        }();
        // One pass over the tiles: each written tile is read once and copied into every view over it, not view by
        // view (speed-16: up to 63 written tiles under 918 views a refresh, ~58,000 per-view lookups and copies,
        // 30-80 ms). A view over a tile that cannot be guarded yet (retry) is left as it is this time; a view not
        // filled yet is filled whole, once.
        marks_.assign(views_.size(), 0);
        written_.clear();
        // The pass over the watched tiles makes a VirtualQuery and a VirtualProtect for each written one (tens of
        // milliseconds to a few hundred under Wine). resolve() waits for mutex_, and a new V# resolved while a
        // submission fixes its console heap slots then waited for the whole pass (speed-35: 116 ms of fixing a
        // submission with indirect draws on). So the lock is let go every 64 tiles. A view resolved meanwhile is
        // filled whole by resolve() and has no mark here; tiles_ is a std::map, so the walk survives its inserts.
        std::size_t walked = 0;
        for (auto& [va, tile] : tiles_) {
            if (!(++walked & 63)) {
                lock.unlock();
                lock.lock();
            }
            const bool written = update(va, tile);
            if (tile.retry)
                for (auto id : tile.views)
                    if (id < marks_.size()) marks_[id] |= kPinned;
            if (written) {
                written_.push_back(&tile);
                written_vas_.push_back(va);
            }
        }
        const std::size_t changed_tiles = written_vas_.size();
        tile_reads_.clear();
        tile_scratch_.clear();
        std::size_t changed_views = 0;
        for (std::size_t w = 0; w < written_vas_.size(); ++w) {
            const Address va = written_vas_[w];
            const Tile& tile = *written_[w];
            const std::uint8_t* read = nullptr;
            bool read_done = false;
            for (const auto id : tile.views) {
                if (id >= marks_.size()) continue;  // made by resolve() during the walk above: filled whole there
                std::uint8_t& mark = marks_[id];
                if (mark & (kPinned | kWhole)) continue;
                View& v = views_[id];
                if (!(mark & kTouched)) {
                    mark |= kTouched;
                    ++changed_views;
                }
                if (whole || !v.ready) {
                    mark |= kWhole;
                    continue;
                }
                if (!read_done) {
                    read = read_tile(va, copy);
                    read_done = true;
                }
                const Address from = (std::max)(va, v.source), to = (std::min)(va + tile_size, v.source + v.size);
                if (from >= to) continue;
                if (read) {
                    std::memcpy(v.dst + (from - v.source), read + (from - va), to - from);
                } else if (copy(v.dst + (from - v.source), reinterpret_cast<const std::uint8_t*>(from), to - from) !=
                           GeometryCopy::copied) {
                    mark |= kWhole;  // filled whole below, which zeroes it if that fails too
                    continue;
                }
                uploaded_ += to - from;
                if (!(mark & kCounted)) {
                    mark |= kCounted;
                    ++uploads_;
                    ++tile_fills_;
                }
            }
        }
        for (std::size_t id = 0; id < marks_.size(); ++id)
            if ((marks_[id] & (kWhole | kPinned)) == kWhole) fill(views_[id], copy);
        written_.clear();
        written_vas_.clear();
        if (uploaded_ != before) ++dirty_refreshes_;
        refilled_ += uploaded_ - before;
        const auto end = std::chrono::steady_clock::now();
        refresh_time_ += end - start;
        // Every 10 s: what the refreshes since the last line did.
        window_.refreshes += 1;
        window_.tiles += changed_tiles;
        window_.views += changed_views;
        window_.bytes += uploaded_ - before;
        window_.time += end - start;
        if (end - window_.since >= std::chrono::seconds(10)) {
            std::fprintf(stderr,
                         "[rt] geometry refresh: %llu refreshes with writes in %.0f s: %.1f written tiles and %.1f views "
                         "each, %.1f KiB copied each, %.3f ms each; %zu tiles watched, %zu views\n",
                         static_cast<unsigned long long>(window_.refreshes),
                         std::chrono::duration<double>(end - window_.since).count(),
                         double(window_.tiles) / double(window_.refreshes),
                         double(window_.views) / double(window_.refreshes),
                         double(window_.bytes) / 1024.0 / double(window_.refreshes),
                         std::chrono::duration<double, std::milli>(window_.time).count() / double(window_.refreshes),
                         tiles_.size(), views_.size());
            window_ = {};
            window_.since = end;
        }
    }

    bool lookup(VkDeviceAddress bda, std::uint64_t& host, std::uint64_t& bytes, std::uint64_t* source = nullptr,
                bool* populated = nullptr) const {
        std::shared_lock lock(structure_mutex_);
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
        std::fprintf(stderr, "[rt] geometry refresh: %llu/%llu submissions uploaded dirty views, %llu bytes (%llu views by "
                             "their written tiles only); %.3f s checking and refreshing sources, %llu bytes cleared\n",
                     static_cast<unsigned long long>(dirty_refreshes_), static_cast<unsigned long long>(submissions_),
                     static_cast<unsigned long long>(refilled_), static_cast<unsigned long long>(tile_fills_),
                     std::chrono::duration<double>(refresh_time_).count(), static_cast<unsigned long long>(cleared_));
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
