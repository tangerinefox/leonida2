"""Build a machine-local DirectStorage predictor sidecar from the player's own pools."""
import argparse
from array import array
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import bcpack_history_calib as calib
import bcpack_tables_bptc as bptc
import gen_bcpack_runtime_tables as pack
import ttpf_mips as ttpf


def fit_slices(pools):
    """Fit same-layer neighbors, including packed 2D mip tails missed by the 256-block scan."""
    arrays = {6: calib.Bc6hCalibrator(), 7: calib.Bc7Calibrator()}
    tails = {6: calib.Bc6hCalibrator(), 7: calib.Bc7Calibrator()}
    for path in pools:
        with closing(ttpf.Pool(path)) as pool:
            if not any(chunk[4] in arrays for chunk in pool.chunks):
                continue
            for index in range(pool.textures):
                d = pool.descriptor(index)
                fit = tails if d["depth"] == 1 else arrays
                codec = next((c for c in fit if d["format"] in calib.CODEC_DXGI[c]), None)
                if not codec or d["dimension"] not in (1, 2):
                    continue
                a = ttpf.analyze(pool, index)
                if not a.get("available") or a.get("blocked"):
                    continue
                layout = a["layout"]
                mips = layout["mips"]
                for layer in range(d["depth"]):
                    def address(m):
                        return (d["base"] + layout.get("data_offset", 0) + m["offset"] +
                                layer * m.get("slice_stride", layout.get("face_size", 0)))
                    for level, mip in enumerate(mips):
                        start = address(mip)
                        chunks = []
                        for virt, size, offset, packed, mode, _ in pool.chunks_for(start, mip["region"]):
                            if mode != codec or start - virt >= 2048 * 16:
                                continue
                            blocks = min(2048, size // 16)
                            if d["depth"] == 1:
                                # In a packed tail the stream starts with other mips or padding.
                                # The usual 2D fit visits only the first 256 physical blocks.
                                first = start + calib.tiled_offset(layout, mip, 0, 0) - virt
                                if not 256 * 16 <= first < blocks * 16:
                                    continue
                            chunks.append((size, offset, packed, virt - start, blocks))
                        if not chunks:
                            continue
                        for neighbor, kind in ((level+1, "child"), (level-1, "parent")):
                            if not 0 <= neighbor < len(mips):
                                continue
                            ref = mips[neighbor]
                            if ref["region"] > 1048576 or pool.codecs(address(ref), ref["region"]) - {0, 8}:
                                continue
                            tiled = pool.read_range(address(ref), ref["region"], {}, bcpack_mode="off")
                            getter = calib._make_sibling_quads_getter(codec, layout, ref, tiled, kind,
                                                                       signed=d["format"] == 96)
                            inverse = calib.make_inverse_tile_lookup(layout, mip)
                            for size, offset, packed, relative, blocks in chunks:
                                lanes = calib.decode_lanes_prefix(pool.data[offset:offset+packed], codec,
                                                                  size, blocks*2)
                                fit[codec].add_pool_chunk(lanes, blocks, inverse, relative, getter)
                            break
    docs = {calib.CODEC_KEYS[c]: f.build_document()[0] for c, f in arrays.items()}
    # Existing array and cube fits keep precedence over supplemental 2D observations.
    extend_missing(docs, {calib.CODEC_KEYS[c]: f.build_document()[0] for c, f in tails.items()})
    return docs


def extend_missing(base, extra):
    for value, flag in (("endpointHistory", "endpointClass"), ("indexHistory", "indexClass")):
        for k, cls in enumerate(extra["bc6h"][flag]):
            if cls and not base["bc6h"][flag][k]:
                base["bc6h"][flag][k] = cls
                base["bc6h"][value][k] = extra["bc6h"][value][k]
    for value, flag in (("historyEndpoints", "historyEndpointBits"), ("historyIndices", "historyIndexBits")):
        for k, row in enumerate(extra["bc7"][flag]):
            for j, depth in enumerate(row):
                if depth and not base["bc7"][flag][k][j]:
                    base["bc7"][flag][k][j] = depth
                    base["bc7"][value][k][j] = extra["bc7"][value][k][j]


def collect(pools, trainer, temporary):
    """The host trainer consumes framed chunks and returns violation counts per candidate."""
    output = temporary / "constraints"
    n = 0
    with output.open("wb") as sink, subprocess.Popen([str(trainer)], stdin=subprocess.PIPE, stdout=sink) as child:
        try:
            with child.stdin:
                for path in pools:
                    with closing(ttpf.Pool(path)) as pool:
                        for _, size, offset, packed, codec, _ in pool.chunks:
                            if codec in (6, 7):
                                child.stdin.write(struct.pack("<III", codec, size, packed))
                                child.stdin.write(pool.data[offset:offset+packed])
                                n += 1
        finally:
            if child.wait():
                raise ValueError("invalid BPTC stream or incomplete constraint pass")
    penalties = array("Q")
    penalties.frombytes(output.read_bytes())
    if sys.byteorder != "little":
        penalties.byteswap()
    if len(penalties) != 2*256*(32+3)*16:
        raise ValueError("truncated trainer output")
    print(f"Scanned {n} BPTC chunks in {len(pools)} pools.", flush=True)
    return penalties


def _compatible_red_fit(before, after, depth, confirmed):
    if (before ^ after) >> (8 - confirmed):
        return False
    return before == after or depth == 8 or (
        4 <= depth < 8 and (before ^ after) >> (8 - depth) == 0)


def refine(docs, votes, history):
    b7 = docs["bc7"]
    effective = bptc.load_bc7(str(history))["history_endpoints"]
    low_fits = {(row, field): value for row, field, value in b7.get("historyEndpointLow2Fits", [])}
    for codec in (6, 7):
        bc6 = codec == 6
        d = docs["bc6h"] if bc6 else b7
        pos = 0 if bc6 else 256*35*16
        for k in range(256):
            for p in range(32):
                penalties = votes[pos:pos+16]
                pos += 16
                if not any(penalties) or (bc6 and p >= 16):
                    continue
                values = d["indexHistory" if bc6 else "historyIndices"][k]
                confirmed = d["indexConfirmedBits" if bc6 else "historyIndexConfirmedBits"][k][p]
                old = values[p]
                candidates = [v for v in range(16) if v >> (4-confirmed) == old >> (4-confirmed)]
                best = min(penalties[v] for v in candidates)
                survivors = [v for v in candidates if penalties[v] == best]
                values[p] = min(survivors, key=lambda v: abs(v-old))
                if not bc6 and best == 0:
                    depth = 4 - (min(survivors) ^ max(survivors)).bit_length()
                    b7["historyIndexBits"][k][p] = max(b7["historyIndexBits"][k][p], depth)
        if bc6:
            continue
        for k in range(256):
            for sub in range(3):
                penalties = votes[pos:pos+16]
                pos += 16
                slots = (8*sub, 8*sub+4)
                old = [effective[k][s] for s in slots]
                choices = []
                for xy, score in enumerate(penalties):
                    pair = ((old[0] & ~3) | xy//4, (old[1] & ~3) | xy%4)
                    # Partial sibling estimates give a leading prefix. Mode-1 parity
                    # constrains the low red bits independently of that prefix.
                    if all(_compatible_red_fit(a, v, b7["historyEndpointBits"][k][s],
                                               b7["historyEndpointConfirmedBits"][k][s])
                           for s, a, v in zip(slots, old, pair)):
                        choices.append((score, sum(abs(a-v) for a,v in zip(old,pair)), pair))
                _, _, pair = min(choices)
                for s, previous, value in zip(slots, old, pair):
                    if b7["historyEndpointBits"][k][s] == 8:
                        b7["historyEndpoints"][k][s] = value
                    elif value != previous:
                        low_fits[k, s] = value & 3
    if low_fits:
        b7["historyEndpointLow2Fits"] = [[k, s, v] for (k, s), v in sorted(low_fits.items())]



def fingerprint(pools, base_history):
    digest = hashlib.sha256()
    for name in ("build_dstorage_history.py", "bcpack_history_train.cpp", "bcpack_history_calib.py",
                 "bcpack.py", "bcpack_tables.py", "bcpack_tables_bptc.py", "gen_bcpack_runtime_tables.py",
                 "ttpf_mips.py", "tex_headers.py", "../runtime/src/bcpack_stream.h"):
        digest.update((TOOLS/name).read_bytes())
    for path in pools:
        stat = os.stat(path)
        digest.update(os.fsencode(os.path.abspath(path)) + struct.pack("<QQ", stat.st_size, stat.st_mtime_ns))
    if base_history:
        for name in ("bc6h.json", "bc7.json"):
            digest.update((base_history/name).read_bytes())
    return digest.hexdigest()


def build(game, output, trainer, base_history=None, json_out=None, if_needed=False):
    pools = calib._resolve_pools([str(game)])
    if any(output.resolve() == Path(p).resolve() for p in pools):
        raise ValueError("sidecar must be separate from its inputs")
    stamp = output.with_suffix(".bcph.stamp")
    signature = fingerprint(pools, base_history)
    if if_needed and stamp.is_file() and output.is_file() and output.stat().st_size == 60936:
        if stamp.read_text(encoding="utf-8") == signature + ":" + hashlib.sha256(output.read_bytes()).hexdigest():
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        temp = Path(tmp)
        print(f"Fitting DirectStorage predictors from {len(pools)} local pools.", flush=True)
        if base_history:
            docs = {name: json.loads((base_history/(name+".json")).read_text(encoding="utf-8")) for name in ("bc6h", "bc7")}
        else:
            fits = {6: calib.Bc6hCalibrator(), 7: calib.Bc7Calibrator()}
            calib.calibrate_from_pools(pools, fits)
            docs = {calib.CODEC_KEYS[c]: f.build_document()[0] for c, f in fits.items()}
        extend_missing(docs, fit_slices(pools))
        history = temp/"history.json"
        history.write_text(json.dumps(docs), encoding="utf-8", newline="\n")
        if not trainer:
            # Cross-built PE helpers cannot run directly on Unix. Build the same small host tool locally.
            tables = temp/"bcpack_tables.inc"
            with tables.open("w", encoding="utf-8", newline="\n") as f:
                pack.generate(f)
            if os.name == "nt":
                trainer = temp/"bcpack_history_train.exe"
                command = ["cl", "/nologo", "/std:c++20", "/EHsc", "/O2", "/I" + str(temp), "/Fe:" + str(trainer)]
            else:
                trainer = temp/"bcpack_history_train"
                command = [os.environ.get("CXX", "c++"), "-std=c++20", "-O3", "-I", str(temp), "-o", str(trainer)]
            subprocess.run([*command, str(TOOLS/"bcpack_history_train.cpp")], cwd=temp, check=True)
        refine(docs, collect(pools, trainer, temp), history)
        ep6 = sum(bool(v) for v in docs["bc6h"]["endpointClass"])
        ix6 = sum(bool(v) for v in docs["bc6h"]["indexClass"])
        ep7 = sum(bool(v) for row in docs["bc7"]["historyEndpointBits"] for v in row)
        ix7 = sum(bool(v) for row in docs["bc7"]["historyIndexBits"] for v in row)
        print(f"Available predictors: BC6H endpoints {ep6}/256, indices {ix6}/256; "
              f"BC7 endpoint fields {ep7}/6144, index fields {ix7}/8192.", flush=True)
        history.write_text(json.dumps(docs), encoding="utf-8", newline="\n")
        pack.write_profile(str(history), temp/"history.bcph")
        signature += ":" + hashlib.sha256((temp/"history.bcph").read_bytes()).hexdigest()
        (temp/"signature").write_text(signature, encoding="utf-8", newline="\n")
        (temp/"history.bcph").replace(output)
        (temp/"signature").replace(stamp)
        if json_out:
            calib.write_outputs(str(json_out), docs, pools)
    print(f"DirectStorage predictor sidecar: {output}", flush=True)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--trainer", type=Path, help="native bcpack_history_train executable")
    ap.add_argument("--base-history", type=Path, help="reuse a local 2D calibration (bc6h.json and bc7.json)")
    ap.add_argument("--if-needed", action="store_true", help="reuse a complete profile for unchanged inputs")
    ap.add_argument("--json-out", type=Path, help="also retain locally fitted JSON for reference checks")
    args = ap.parse_args()
    build(args.game, args.out, args.trainer, args.base_history, args.json_out, args.if_needed)
