"""Two synthetic compute records exercise parallel translation, cache reuse and interrupted publication."""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("spirv-val") and shutil.which("spirv-as"),
                     "requires Linux and SPIRV-Tools")
class ComputeBatch(unittest.TestCase):
    def test_resume_and_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "input", root / "output"
            (src / "disasm").mkdir(parents=True)
            (src / "fspx").mkdir()
            rows = []
            for i in (0, 1):
                name = f"cs_test_{i}"
                rows.append({"file": name + ".dxbc", "stage": "cs"})
                code = struct.pack("<II", 0xbf800000 | i, 0xbf810000)
                (src / "fspx" / (name + ".code")).write_bytes(code)
                (src / "fspx" / (name + ".state")).write_bytes(struct.pack("<7I", 0, 0, 0, 1, 1, 0, 0))
                (src / "fspx" / (name + ".json")).write_text(
                    json.dumps({"stage": "cs", "threads": [1, 1, 1], "rsrc2": 0, "wave32": True}),
                    encoding="utf-8", newline="\n")
                (src / (name + ".dxbc")).write_bytes(b"DXBC" + bytes(28))
                listing = f"; {name}  code 0x0..0x8 (fspx), gfx1013, 0 invalid words\n"
                for asm, word in ((f"s_nop {i}", code[:4]), ("s_endpgm", code[4:])):
                    listing += f"\t{asm} ; encoding: [" + ",".join(hex(b) for b in word) + "]\n"
                (src / "disasm" / (name + ".s")).write_text(listing, encoding="utf-8", newline="\n")
            (src / "index.json").write_text(json.dumps(rows), encoding="utf-8", newline="\n")
            cmd = [sys.executable, "-I", str(ROOT / "scripts/translate_pipeline_compute.py"),
                   str(src), "--out", str(out), "--wave64-emulate"]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            index = json.loads((out / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index), 2)
            files = sorted(out.glob("*.spv"))
            self.assertEqual(len(files), 2)
            for p in files:
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), p.stem.rsplit("_", 1)[1])
                os.utime(p, ns=(1000000000, 1000000000))
            (out / "index.json").unlink()  # interruption after checkpointing both shaders
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(index, json.loads((out / "index.json").read_text(encoding="utf-8")))
            self.assertTrue(all(p.stat().st_mtime_ns == 1000000000 for p in files))
            files[0].write_bytes(b"damaged")
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(hashlib.sha256(files[0].read_bytes()).hexdigest(), files[0].stem.rsplit("_", 1)[1])
            self.assertEqual(files[1].stat().st_mtime_ns, 1000000000)
            # A newly rejected input must leave the current index, even when the older file remains readable.
            (src / "disasm/cs_test_0.s").write_text("; synthetic empty listing\n", encoding="utf-8", newline="\n")
            partial = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(partial.returncode, 1, partial.stderr)
            self.assertEqual(len(json.loads((out / "index.json").read_text(encoding="utf-8"))), 1)
            failures = json.loads((out / "failures.json").read_text(encoding="utf-8"))
            rejected = next(m for n, m in index.items() if n.startswith("cs_test_0_"))
            self.assertIn("no instructions", failures[rejected["fspx_code_sha256"]][rejected["fspx_state_sha256"]])


if __name__ == "__main__":
    unittest.main()
