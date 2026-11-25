"""Per-lane readback for terminating and unreachable loop bounds."""
from pathlib import Path
import unittest
from unittest.mock import patch

from test_loop_ballot_barrier import _translate_cs, rdna2spirv
from spirv_mini_interp import run_workgroup

FIXTURE = Path(__file__).with_name("fixtures") / "loop_iteration_guard.asm"


def module(lanes, limit, structured=True):
    lines = [s for s in FIXTURE.read_text().splitlines() if s and not s.startswith(";")]
    lines.insert(0, "s_mov_b64 exec, -1" if lanes == 64 else "s_mov_b32 exec_lo, -1")
    with patch.object(rdna2spirv, "LOOP_LIMIT", limit):
        text, gen = _translate_cs("cs_guard", lines, threads=lanes, emulate=lanes == 64, structured=structured)
    assert gen.wave64 == (lanes == 64) and gen.emulate == (lanes == 64)
    return text


def execute(text, lanes, bound):
    mem = run_workgroup(
        text, lanes, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                              "%sg_lane": i % 32, "%local_index": i},
        push_constants=[0, 0, lanes * 4, 0x31016FAC, bound] + [0] * 27, memory={})
    return [mem[i * 4] for i in range(lanes)]


class LoopGuardTests(unittest.TestCase):
    def test_finite_and_nonterminating(self):
        for lanes in (32, 64):
            with self.subTest(lanes=lanes):
                plain, guarded = module(lanes, 0), module(lanes, 16)
                self.assertNotIn("%loop_guard_", plain)
                self.assertIn("%loop_guard_", guarded)
                self.assertEqual(execute(plain, lanes, 6), [6] * lanes)
                self.assertEqual(execute(guarded, lanes, 6), [6] * lanes)
                self.assertEqual(execute(guarded, lanes, 1), [32] * lanes)

    def test_dispatch_loop_also_stops(self):
        for lanes in (32, 64):
            with self.subTest(lanes=lanes):
                guarded = module(lanes, 16, structured=False)
                self.assertEqual(execute(guarded, lanes, 6), [6] * lanes)
                words = execute(guarded, lanes, 1)
                self.assertTrue(all(0 < v <= 32 and v % 2 == 0 for v in words))


    def test_inner_budget_resets_on_each_entry(self):
        lines = ["s_mov_b32 s8, 0", "s_mov_b32 s9, 0", "outer:", "s_mov_b32 s10, 0",
                 "inner:", "s_add_u32 s9, s9, 1", "s_add_u32 s10, s10, 1", "s_cmp_eq_u32 s10, 3",
                 "s_cbranch_scc0 inner", "s_add_u32 s8, s8, 1", "s_cmp_eq_u32 s8, 3", "s_cbranch_scc0 outer",
                 "v_mov_b32_e32 v1, s9", "v_lshlrev_b32_e32 v2, 2, v0",
                 "buffer_store_dword v1, v2, s[0:3], 0 offen", "s_endpgm"]
        with patch.object(rdna2spirv, "LOOP_LIMIT", 3):
            text, _ = _translate_cs("cs_guard_nested", lines, threads=32)
        self.assertEqual(execute(text, 32, 0), [9] * 32)
