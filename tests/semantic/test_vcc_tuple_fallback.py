"""A scalar tuple that runs into vcc is translated from the blob when llvm-mc cannot decode it.

llvm-mc rejects s_buffer_load_dwordx8 into s[100:107], a buffer resource in s[104:107] and an image resource in
s[100:107] (SGPRs 106 and 107 are vcc). It skips the first word as invalid and leaves the second word as junk, so
translate() refused every such shader ("reachable instruction with normalized encoding", or "fall-through into
undecodable words"; docs/shader_library.md section 5), or, when the junk matched the blob, translated a program
with the load missing. The words are now decoded from the blob as the VCCTuple lists the translator takes.

Hand-built listings and blobs of synthetic instructions (encodings from llvm-mc 19, the three tuple words patched in
their register field), no game data.

    python -I tests/semantic/test_vcc_tuple_fallback.py -v
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import rdna2spirv, translate_listing  # noqa: E402
import fssm_header  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

NAME = "cs_vcc_tuple"
CODE_START = 0x148
BASE, OUT = 0x1000, 0x3000
INVALID = "; warning: invalid instruction encoding"

SET_SGPR_V = [  # s4..s7 = a V# over BASE (32 bytes), as tests/semantic/test_fspx_rules.py builds it
    ("s_mov_b32 s4, 0x1000", "ff 03 84 be 00 10 00 00"),
    ("s_mov_b32 s5, 0", "80 03 85 be"),
    ("s_mov_b32 s6, 32", "a0 03 86 be"),
    ("s_mov_b32 s7, 0x30000000", "ff 03 87 be 00 00 00 30"),
]
WAIT = ("s_waitcnt lgkmcnt(0)", "7f c0 8c bf")
WAIT_V = ("s_waitcnt vmcnt(0)", "70 3f 8c bf")
STORE_X4 = [
    ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
    ("global_store_dwordx4 v6, v[10:13], s[0:1]", "00 80 78 dc 06 0a 00 00"),
    ("s_endpgm", "00 00 81 bf"),
]
# the words llvm-mc cannot decode (each is two dwords): the llvm-mc 19 encoding of the same instruction one tuple
# lower, with its register field patched into vcc
LOAD_X8_VCC = "02 19 2c f4 00 00 00 fa"        # s_buffer_load_dwordx8 [s100..s105,vcc_lo,vcc_hi], s[4:7], 0x0
LOAD_X4_VCC = "02 1a 28 f4 00 00 00 fa"        # s_buffer_load_dwordx4 [s104,s105,vcc_lo,vcc_hi], s[4:7], 0x0
LOAD_DW_FROM_VCC = "b4 02 20 f4 00 00 00 fa"   # s_buffer_load_dword s10, [s104,s105,vcc_lo,vcc_hi], 0x0
BUFFER_LOAD_VCC = "00 10 30 e0 01 0b 1a 80"    # buffer_load_dword v11, v1, [s104,s105,vcc_lo,vcc_hi], 0 offen
IMAGE_LOAD_VCC = "08 1f 00 f0 01 04 19 00"     # image_load v[4:7], v[1:2], [s100..s105,vcc_lo,vcc_hi] 2D unorm
PROGRAM_VARIANTS = ("invalid", "junk", "normalized")


def encoding(raw):
    return "; encoding: [" + ",".join(f"0x{b:02x}" for b in bytes.fromhex(raw)) + "]"


def listing_and_blob(program, variant="invalid"):
    """program: (text, hex) pairs; a pair whose text is None is a word llvm-mc rejects, listed as the variant says."""
    code = b"".join(bytes.fromhex(raw) for _, raw in program)
    lines = []
    for text, raw in program:
        if text is not None:
            lines.append(f"\t{text:<40}{encoding(raw)}")
            continue
        second = bytes.fromhex(raw)[4:8]
        lines.append(INVALID)
        if variant == "invalid":
            lines.append(INVALID)
        elif variant == "junk":         # the second word decodes as an instruction of its own, bytes as stored
            lines.append(f"\ts_nop 0{'':<33}" + encoding(second.hex()))
        else:                           # printed with another encoding than the blob holds
            lines.append("\texp mrt0 v0, v0, v0, v0 done vm " + encoding("0f 18 00 f8 00 00 00 00"))
    head = f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + len(code):x} (scan), gfx1013, 2 invalid words"
    header = bytearray(CODE_START)
    struct.pack_into("<4I", header, 0x138, 0x88, 32, 1, 1)
    struct.pack_into("<I", header, fssm_header.WAVE_FLAGS_OFFSET, 0x20)       # wave32
    return ((head + "\n" + "\n".join(lines) + "\n").encode(), bytes(header) + code)


def translate(program, variant="invalid"):
    data, blob = listing_and_blob(program, variant)
    return translate_listing(NAME, data, fssm=blob)


def run(spv, memory):
    mem = dict(memory)
    si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


class VccTupleFallbackTest(unittest.TestCase):
    def test_load_into_a_tuple_ending_in_vcc(self):
        program = SET_SGPR_V + [
            (None, LOAD_X8_VCC), WAIT,
            ("v_mov_b32_e32 v10, s100", "64 02 14 7e"), ("v_mov_b32_e32 v11, s105", "69 02 16 7e"),
            ("v_mov_b32_e32 v12, vcc_lo", "6a 02 18 7e"), ("v_mov_b32_e32 v13, vcc_hi", "6b 02 1a 7e"),
        ] + STORE_X4
        words = [0x11110000 + i for i in range(8)]
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                mem = run(translate(program, variant), {BASE + 4 * i: w for i, w in enumerate(words)})
                self.assertEqual([mem.get(OUT + 4 * i) for i in range(4)], [words[0], words[5], words[6], words[7]])

    def test_buffer_resource_in_vcc(self):
        # a V# loaded into [s104,s105,vcc_lo,vcc_hi], then read by s_buffer_load and by buffer_load
        program = SET_SGPR_V + [
            (None, LOAD_X4_VCC), WAIT,
            (None, LOAD_DW_FROM_VCC), WAIT,
            ("v_mov_b32_e32 v1, 4", "84 02 02 7e"),
            (None, BUFFER_LOAD_VCC), WAIT_V,
            ("v_mov_b32_e32 v10, s10", "0a 02 14 7e"),
            ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
            ("global_store_dwordx2 v6, v[10:11], s[0:1]", "00 80 74 dc 06 0a 00 00"),
            ("s_endpgm", "00 00 81 bf"),
        ]
        memory = {BASE + 4 * i: w for i, w in enumerate([BASE + 0x100, 0, 16, 0x30000000])}
        memory[BASE + 0x100], memory[BASE + 0x104] = 0xCAFE1234, 0xBEEF5678
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                mem = run(translate(program, variant), memory)
                self.assertEqual([mem.get(OUT), mem.get(OUT + 4)], [0xCAFE1234, 0xBEEF5678])

    def test_image_resource_in_vcc(self):
        program = SET_SGPR_V + [
            (None, LOAD_X8_VCC), WAIT,
            ("v_mov_b32_e32 v1, 0", "80 02 02 7e"), ("v_mov_b32_e32 v2, 0", "80 02 04 7e"),
            (None, IMAGE_LOAD_VCC), WAIT_V,
            ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
            ("global_store_dword v6, v4, s[0:1]", "00 80 70 dc 06 04 00 00"),
            ("s_endpgm", "00 00 81 bf"),
        ]
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                self.assertIn("OpFunctionCall %uint %desc_lookup_fn", translate(program, variant))

    def test_other_undecodable_words_are_still_refused(self):
        # SHADER-813: a rejected word that is not a tuple into vcc (sdata 112, not a register) stays undecodable,
        # and the code that runs into it is refused
        word = "02 1c 2c f4 00 00 00 fa"
        program = SET_SGPR_V + [(None, word)] + STORE_X4
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "fall-through into undecodable words"):
            translate(program)

    def test_nsa_mimg_with_vcc_tuple_does_not_read_past_code_end(self):
        # A 12-byte NSA1 image_load whose 3rd dword lies past code_end must not be restored across stop.
        mimg_nsa1 = bytes.fromhex("0a 1f 00 f0 01 04 19 00 02 00 00 00")
        self.assertEqual(rdna2spirv.restore_skipped_instructions([], mimg_nsa1, 0, 8), [])
        restored = rdna2spirv.restore_skipped_instructions([], mimg_nsa1, 0, 12)
        self.assertEqual(len(restored), 1)
        self.assertEqual((restored[0]["pc"], len(restored[0]["enc"]) // 2), (0, 12))


if __name__ == "__main__":
    unittest.main()
