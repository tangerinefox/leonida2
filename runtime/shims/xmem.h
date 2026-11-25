#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rt::xmem {

void geometry_aliases(std::uintptr_t va, std::vector<std::uintptr_t>& addresses);
bool commit_physical_range(const void* address, std::size_t size);
bool commit_physical_pages(const std::uint64_t* page_array, std::size_t count);

// Copy a validated DirectStorage staging buffer through the page table. The page list has
// one frame number per 64 KiB page. All pages are resolved before a write starts.
bool transfer_physical(std::span<const std::uint64_t> pages, std::uint16_t offset,
                       std::span<std::uint8_t> staging, bool write);

// Console queue slot 31, CopyPageMappingsBatchX (d3d12_xs_device.cpp): maps at `destination` the physical pages that
// are mapped at `source` now, `pages` 64 KiB pages one after another, or the one page at `source` every time with
// `repeat_first`. A source page with nothing mapped leaves its destination page unmapped; a null `source` unmaps them
// all. False, with the last error set, for an unaligned address or a destination outside a physical-page window.
// Set by xmem.cpp when it is linked: the command-object proxy is also compiled into tests without it.
using CopyPageMappingsFn = bool (*)(void* destination, const void* source, std::uint64_t pages, bool repeat_first);
inline CopyPageMappingsFn copy_page_mappings_fn = nullptr;

}
