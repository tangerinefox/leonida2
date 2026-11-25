"""Build an interrupted or new pipeline compute pack from shader_library.py output (Linux).

Two translation processes, including the parent, each run at most one SPIRV-Tools process.
Validated modules and their receipts survive interruption. The runtime index is published last.
"""
import argparse
import fcntl
import hashlib
import json
import os
import resource
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
from make_game import file_sha256 as digest, translation_sources_digest
import rdna2spirv as translator


def build_one(src, out, name, settings, compact=False):
    listing = src / "disasm" / (name + ".s")
    side = src / "fspx" / name
    code, state = (Path(str(side) + suffix) for suffix in (".code", ".state"))
    code_key, state_key = digest(code), digest(state)
    container = src / (name + ".dxbc")
    inputs = [settings, name, code_key, state_key, digest(listing), digest(Path(str(side) + ".json")),
              digest(container) if container.is_file() else None]
    key = hashlib.sha256(json.dumps(inputs).encode()).hexdigest()
    receipt = out / ".compute-cache" / (key + ".json")
    if receipt.exists():
        try:
            row = json.loads(receipt.read_text(encoding="utf-8"))
            if (row["meta"]["fspx_code_sha256"], row["meta"]["fspx_state_sha256"]) == (code_key, state_key):
                module_hash = row["module"]
                if len(module_hash) == 64 and all(c in "0123456789abcdef" for c in module_hash):
                    if digest(out / (name + "_" + module_hash + ".spv")) == module_hash:
                        return row
        except (OSError, ValueError, KeyError, TypeError):
            pass

    asm = translator.translate(str(listing))
    with tempfile.TemporaryDirectory(dir=out / ".compute-cache") as tmp:
        sa, sv = Path(tmp) / "module.spvasm", Path(tmp) / "module.spv"
        sa.write_text(asm, encoding="utf-8", newline="\n")
        result = translator.assemble_spirv(str(sa), str(sv), asm_text=asm, compact=compact)
        if result.returncode:
            raise translator.Unsupported((result.stderr.splitlines() or ["SPIR-V tool failed"])[0])
        meta = translator.shader_meta(translator.translate.last_gen, asm, src_path=str(listing))
        if (meta.get("stage"), meta.get("fspx_code_sha256"), meta.get("fspx_state_sha256")) != (
                "cs", code_key, state_key):
            raise ValueError("compute stage or FSPX identity mismatch")
        module_hash = digest(sv)
        sv.replace(out / (name + "_" + module_hash + ".spv"))
    row = {"meta": meta, "module": module_hash}
    translator.write_index(str(receipt), row)
    return row


def build_part(src, out, names, settings, report, compact=False):
    rows, failures = {}, {}
    for number, name in enumerate(names, 1):
        try:
            rows[name] = build_one(src, out, name, settings, compact)
        except (translator.Unsupported, MemoryError) as e:
            failures[name] = str(e) or type(e).__name__
        if number % 100 == 0 or number == len(names):
            print(f"worker {os.getpid()}: {number}/{len(names)}, {len(failures)} failed", flush=True)
    translator.write_index(str(report), {"rows": rows, "failures": failures})


def build(src, out, names, settings, jobs, compact=False):
    cache = out / ".compute-cache"
    cache.mkdir(exist_ok=True)
    with (cache / "lock").open("ab") as lock, tempfile.TemporaryDirectory(dir=cache) as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reports = [Path(tmp) / str(i) for i in range(jobs)]
        pid = os.fork() if jobs == 2 else None
        if pid == 0:
            try:
                build_part(src, out, names[1::2], settings, reports[1], compact)
            except BaseException:
                import traceback
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        try:
            build_part(src, out, names[::jobs], settings, reports[0], compact)
        finally:
            if pid and os.waitpid(pid, 0)[1]:
                raise RuntimeError("second translation worker failed")

        index, failures, pairs = {}, {}, {}
        for report in reports:
            part = json.loads(report.read_text(encoding="utf-8"))
            failures.update(part["failures"])
            for name, row in part["rows"].items():
                module, meta = row["module"], row["meta"]
                pair = meta["fspx_code_sha256"], meta["fspx_state_sha256"]
                if pairs.setdefault(pair, module) != module:
                    raise ValueError("conflicting compute modules for one code/state pair")
                # The same binary can implement different code/state keys. Each needs a runtime entry.
                entry = name + "_" + module
                index[entry] = meta
        by_hash = {}
        for name, reason in failures.items():
            code = digest(src / "fspx" / (name + ".code"))
            state = digest(src / "fspx" / (name + ".state"))
            by_hash.setdefault(code, {})[state] = reason
        translator.write_index(str(out / "failures.json"), by_hash)
        translator.write_index(str(out / "index.json"), index)
        print(f"{len(index)}/{len(names)} valid, {len(failures)} failed")
        return int(bool(failures))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", type=Path, help="unpacked library with index.json, disasm/ and fspx/")
    ap.add_argument("--out", type=Path, required=True, help="dedicated runtime library subdirectory")
    ap.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    ap.add_argument("--memory-mib", type=int, default=768, help="virtual memory per process (default 768 MiB)")
    ap.add_argument("--descriptor-hash", choices=translator.DESCRIPTOR_HASHES, default="fnv1a",
                    help="bindless descriptor table hash shared with the runtime")
    ap.add_argument("--loop-limit", type=translator.spirv_loop_guard.parse_limit,
                    default=translator.spirv_loop_guard.DEFAULT_LIMIT, help="maximum loop visits (0 disables)")
    ap.add_argument("--compact-spirv", action="store_true", help="compact modules with spirv-opt before validation")
    ap.add_argument("--wave64-emulate", action="store_true")
    args = ap.parse_args()
    if args.memory_mib < 1:
        ap.error("--memory-mib must be positive")
    src, out = args.inputs.resolve(), args.out.resolve()
    if out == src or src in out.parents or out in src.parents:
        ap.error("input and output directories must be separate")
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    bound = min(n for n in (args.memory_mib << 20, soft, hard) if n != resource.RLIM_INFINITY)
    resource.setrlimit(resource.RLIMIT_AS, (bound, hard))
    versions = []
    required_tools = ["spirv-as", "spirv-val"]
    if args.compact_spirv:
        required_tools.append("spirv-opt")
    for tool in required_tools:
        binary = shutil.which(tool)
        if not binary:
            ap.error(tool + " is required")
        versions.append(digest(Path(binary)))
    settings = [translation_sources_digest(), digest(Path(__file__)), versions, args.wave64_emulate,
                args.loop_limit, args.descriptor_hash]
    rows = json.loads((src / "index.json").read_text(encoding="utf-8"))
    names = []
    for row in rows:
        if row["stage"] != "cs":
            continue
        filename = row["file"]
        if Path(filename).name != filename or not filename.endswith(".dxbc") or "\\" in filename:
            ap.error("invalid compute filename in input index")
        names.append(filename[:-5])
    if not names or len(set(names)) != len(names):
        ap.error("expected unique compute filenames in the input index")
    out.mkdir(parents=True, exist_ok=True)
    translator.DESCRIPTOR_HASH = args.descriptor_hash
    translator.LOOP_LIMIT = args.loop_limit
    translator.WAVE64_EMULATE = args.wave64_emulate
    return build(src, out, sorted(names), settings, args.jobs, args.compact_spirv)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    sys.exit(main())
