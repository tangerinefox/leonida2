// Temporary payloads across all DirectStorage queues. A pair reserves both requests
// together so its ordered publisher never waits for space held by its own helper.
#pragma once

#include "dstorage_xs_abi.h"
#include "../src/dstorage_trace.h"
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace leonida::dstorage {

inline uint64_t staging_bytes(const dsx::DSTORAGE_REQUEST& r) {
    using namespace dsx;
    const auto o = r.Options.AsUINT64;
    const bool bc = options::bcpack(o) != 0;
    uint64_t input = r.SourceSize, peak = 0;
    const auto pages = [](uint64_t size, unsigned offset) {
        return size ? size + sizeof(uint64_t) * (1 + (size - 1 + offset) / 65536) : 0;
    };
    if (o & options::kSourcePages) peak = pages(input, r.SourcePageOffset);
    const auto stage = [&](uint64_t output) {
        peak = (std::max)(peak, input + output);
        input = output;
    };
    if (options::zlib(o)) stage(bc ? r.IntermediateSize : r.DestinationSize);
    if (bc) stage(r.DestinationSize);
    if (options::swizzle(o)) stage(r.DestinationSize);
    if (o & options::kDestinationPages) peak = (std::max)(peak, pages(input, r.DestinationPageOffset));
    return peak; // Raw buffer reads/copies allocate no private payload.
}

class StagingPool {
public:
    static constexpr uint64_t limit = 64ull << 20;
    // Requests above the shared window run alone. The validated ABI sizes bound
    // one transform to two 1 GiB buffers. No payload is allocated by the pool itself.
    class Lease {
    public:
        explicit Lease(StagingPool& pool) : pool_(pool) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        bool try_pair(uint64_t bytes) {
            std::lock_guard lock(pool_.mutex_);
            if (pool_.helpers_ >= 2 || pool_.next_ != pool_.serving_ ||
                bytes > limit || pool_.used_ > limit - bytes) return false;
            ++pool_.helpers_;
            pair_ = true;
            take(bytes);
            return true;
        }
        void acquire(uint64_t bytes) {
            if (!bytes) return;
            std::unique_lock lock(pool_.mutex_);
            const uint64_t ticket = pool_.next_++;
            pool_.changed_.wait(lock, [&] {
                return ticket == pool_.serving_ && (!pool_.used_ ||
                       (bytes <= limit && pool_.used_ <= limit - bytes));
            });
            take(bytes);
            ++pool_.serving_;
            if (pool_.next_ != pool_.serving_) pool_.changed_.notify_all();
        }
        ~Lease() {
            if (!bytes_ && !pair_) return;
            std::lock_guard lock(pool_.mutex_);
            pool_.used_ -= bytes_;
            if (pair_) --pool_.helpers_;
            if (!pool_.used_ && pool_.drained_ < pool_.peak_) {
                pool_.drained_ = pool_.peak_;
                if (trace_enabled()) std::fprintf(stderr, "[rt] ds-staging idle_bytes=0 peak_bytes=%llu\n", pool_.peak_);
            }
            if (pool_.next_ != pool_.serving_) pool_.changed_.notify_all();
        }
    private:
        void take(uint64_t bytes) {
            bytes_ = bytes;
            pool_.used_ += bytes;
            if (pool_.used_ > pool_.peak_) {
                pool_.peak_ = pool_.used_;
                if (trace_enabled()) std::fprintf(stderr, "[rt] ds-staging peak_bytes=%llu limit_bytes=%llu\n", pool_.peak_, limit);
            }
        }
        StagingPool& pool_;
        uint64_t bytes_{};
        bool pair_{};
    };

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    unsigned helpers_{};
    uint64_t used_{}, peak_{}, drained_{}, next_{}, serving_{};
};

inline StagingPool& staging_pool() {
    static auto* pool = new StagingPool;
    return *pool;
}

} // namespace leonida::dstorage
