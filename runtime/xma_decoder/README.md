# XMA2 decoder C ABI

This crate exposes a C ABI around the separately licensed xma Rust package pinned in Cargo.toml. It decodes one mono or stereo stream from a non-empty buffer of complete 2048-byte XMA2 packets into interleaved signed 16-bit PCM.

Each decoder handle serializes decode calls. The returned PCM block owns its samples and stays valid until leonida_xma2_pcm_destroy, even after its decoder is destroyed. Do not destroy a decoder concurrently with a decode call or destroy a handle more than once.

The tests build packets in memory; they do not open or contain audio fixtures. The synthetic C++ smoke test verifies the exported ABI and ownership boundary.

Run:

```powershell
cargo fmt --manifest-path runtime/xma_decoder/Cargo.toml -- --check
cargo test --manifest-path runtime/xma_decoder/Cargo.toml --locked
cargo build --manifest-path runtime/xma_decoder/Cargo.toml --locked
```

The C++ smoke test is run by the xma2-decoder workflow with MSVC. It links ws2_32, userenv, and ntdll for Rust's Windows standard library.

## Use in the runtime

`runtime/CMakeLists.txt` builds this crate for `x86_64-pc-windows-msvc` at configure time when `cargo` is on `PATH` and links it into `gta6rt` and `acphal_test` with `LEONIDA_HAVE_XMA2_DECODER` defined. The first build fetches the pinned `xma` git dependency. Set `CARGO_NET_OFFLINE=true` for a configure that uses only the local Cargo cache.

`runtime/shims/acphal.cpp` uses `leonida_xma2_decoder_create_hardware` and decodes whole 2048-byte packets, so loop points are rounded to packets. The shim creates a fresh decoder after a loop rewind or a decoder error.

Without `cargo`, when that build fails, or with the cl.exe fallback of `scripts/build_runtime.ps1`, XMA voices output silence.
Use `-DLEONIDA_REQUIRE_XMA2_DECODER=ON` to require the decoder in a CMake build, as the Windows runtime CI does.
