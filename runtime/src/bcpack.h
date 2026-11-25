// C++ BCPack decoder. Publish the caller's staged output only on success.
#pragma once
#include <array>
#include <cstdint>
#include <span>

namespace leonida::bcpack {
enum class Quality { failed, confirmed, fitted };

// Fixed-size little-endian sidecar produced by
// tools/gen_bcpack_runtime_tables.py --profile. All fields are bytes: BC6H's 12
// endpoint quantities are stored as little-endian uint16 pairs.
struct Calibration {
  struct BC6 {
    std::array<std::uint8_t, 24> ep;
    std::array<std::uint8_t, 16> ix;
    std::array<std::uint8_t, 12> ep_confirmed;
    std::array<std::uint8_t, 16> ix_confirmed;
    std::uint8_t ep_class, ix_class;
  };
  struct BC7 {
    std::array<std::uint8_t, 24> ep, ep_bits, ep_confirmed;
    std::array<std::uint8_t, 32> ix, ix_bits, ix_confirmed;
  };
  std::array<BC6, 256> bc6;
  std::array<BC7, 256> bc7;
};
static_assert(sizeof(Calibration) == 256 * (70 + 168));
Quality decode(unsigned codec, std::span<const std::uint8_t> source,
               std::span<std::uint8_t> output, bool allow_fitted = false,
               const Calibration *history = nullptr, const char** failure = nullptr,
               bool fast_entropy = true);
} // namespace leonida::bcpack
