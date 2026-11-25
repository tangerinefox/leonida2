// leonida::kernel_writes::read_file (runtime/src/title_page_watch.h) into memory that is reserved until its first
// touch, as xmem maps the title's physical pages (SEC_RESERVE section views committed by a fault handler): the kernel's
// write does not fault such a page in, so the read fails with ERROR_NOACCESS unless the committer runs first. No game
// data.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/title_page_watch.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

constexpr DWORD kViewBytes = 1 << 20;
constexpr DWORD kFileBytes = 256 * 1024;

std::uint8_t* g_view = nullptr;
int g_commits = 0;

// Stands in for rt::xmem::commit_physical_range: commits the part of the test view the range covers.
bool commit_view(const void* address, std::size_t bytes) {
    ++g_commits;
    const auto* begin = static_cast<const std::uint8_t*>(address);
    if (begin < g_view || begin + bytes > g_view + kViewBytes) return true;  // not ours
    return VirtualAlloc(const_cast<std::uint8_t*>(begin), bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

}  // namespace

int main() {
    wchar_t dir[MAX_PATH], path[MAX_PATH];
    GetTempPathW(MAX_PATH, dir);
    GetTempFileNameW(dir, L"kwt", 0, path);
    std::vector<std::uint8_t> bytes(kFileBytes);
    for (DWORD i = 0; i < kFileBytes; ++i) bytes[i] = static_cast<std::uint8_t>(i * 7 + (i >> 9));
    HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    DWORD written = 0;
    check(file != INVALID_HANDLE_VALUE && WriteFile(file, bytes.data(), kFileBytes, &written, nullptr) &&
              written == kFileBytes, "temporary file written");

    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, kViewBytes,
                                        nullptr);
    g_view = section ? static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kViewBytes))
                     : nullptr;
    MEMORY_BASIC_INFORMATION info{};
    check(g_view && VirtualQuery(g_view, &info, sizeof(info)) && info.State == MEM_RESERVE,
          "the destination view is reserved, not committed");
    std::uint8_t* destination = g_view + 0x10000;

    // Without a committer: what the title saw (Windows answers ERROR_NOACCESS; another host may fault the page in).
    OVERLAPPED at{};
    DWORD got = 0;
    const BOOL plain = leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at);
    std::printf("   without a committer: %s, error %lu\n", plain ? "read" : "failed", plain ? 0ul : GetLastError());

    leonida::kernel_writes::set_committer(&commit_view);
    at = {};
    got = 0;
    const BOOL ok = leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at);
    check(ok && got == kFileBytes, "with the committer the read completes");
    check(ok && std::memcmp(destination, bytes.data(), kFileBytes) == 0, "and the destination holds the file's bytes");
    check(plain || g_commits == 1, "the committer ran once, for the failed read's retry");

    // A read that succeeds first time does not call the committer.
    const int before = g_commits;
    at = {};
    check(leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at) && g_commits == before,
          "a read into committed memory does not call the committer");

    leonida::kernel_writes::set_committer(nullptr);
    UnmapViewOfFile(g_view);
    CloseHandle(section);
    CloseHandle(file);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
