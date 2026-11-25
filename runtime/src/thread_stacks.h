#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace rt {

// One thread's call chain, taken without stopping the thread for longer than a context and stack copy.
class StackSampler {
public:
    static constexpr std::size_t kMaxFrames = 40;
    static constexpr std::size_t kMaxStackBytes = 8 * 1024 * 1024;

    StackSampler();  // Allocates the stack copy once, so sample() allocates nothing.

    // `thread` needs THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT. While it is suspended this makes system calls only
    // (GetThreadContext, VirtualQuery, ReadProcessMemory): no heap, stdio or user-mode locks, which the suspended
    // thread may hold. After ResumeThread it unwinds the copy with RtlLookupFunctionEntry/RtlVirtualUnwind, reading
    // stack words from the copy rather than the live stack, which has moved on. Writes up to kMaxFrames addresses
    // (the instruction pointer, then return addresses) and returns how many; 0 if the thread could not be sampled.
    std::size_t sample(void* thread, std::uint64_t* frames);

private:
    std::unique_ptr<std::byte[]> stack_;
};

// Parses LEONIDA_STACKS: comma-separated whole seconds. Returns false (and leaves `seconds` empty) on bad text.
bool parse_stack_times(std::string_view text, std::vector<unsigned>& seconds);

// LEONIDA_STACKS: snapshots of every thread at the listed times since this call, printed to stderr. Destroy the
// sampler before unmapping the title image.
std::jthread start_thread_stacks();

}  // namespace rt
