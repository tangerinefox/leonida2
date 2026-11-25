"""Build C++ codec tables from the shipped Python decoder. Reads no user input."""
import argparse
from pathlib import Path
import sys
import struct

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bcpack as bc
import bcpack_tables as tables
import bcpack_tables_bptc as bptc


def initializer(values):
    if isinstance(values, int):
        return str(values)
    return "{" + ", ".join(initializer(v) for v in values) + "}"


def emit(out, ctype, name, rows):
    print(f"constexpr {ctype} {name} = {{", file=out)
    for row in rows:
        print("    " + initializer(row) + ",", file=out)
    print("};", file=out)


def generate(out):
    emit(out, "uint32_t", "freq[11][16][11]", [
        [tables.PROB_TABLES.get((w, q), ()) for q in range(16)] for w in range(11)
    ])
    emit(out, "int", "widths[8][16]", [bc.LANE_WIDTHS.get(c, ()) for c in range(8)])
    for codec in (1, 3, 4, 5):
        known = getattr(tables, f"BC{codec}_HISTORY_CONFIRMED")
        fitted = getattr(tables, f"BC{codec}_HISTORY_PROVISIONAL")
        emit(out, "Seed", f"history{codec}[256]", [
            (known.get(i, fitted.get(i, (-1,))), int(i not in known and i in fitted)) for i in range(256)
        ])
    for name, rows in (("quad1", tables.BC1_QUADRANT), ("quad3", tables.BC3_COLOUR_QUADRANT),
                       ("pair3", tables.BC3_ALPHA_PAIR), ("pair4", tables.BC4_PAIR), ("pair5", tables.BC5_PAIR)):
        values = [v if isinstance(v, int) else v[0] | (v[1] << 3) for v in rows]
        print(f"constexpr uint8_t {name}[] = {initializer(values)};", file=out)
    for name, rows in (("anchor2", bptc.BC7_ANCHOR2), ("anchor3a", bptc.BC7_ANCHOR3A),
                       ("anchor3b", bptc.BC7_ANCHOR3B)):
        print(f"constexpr uint8_t {name}[] = {initializer(rows)};", file=out)
    cfg = bptc.load_bc7()
    emit(out, "uint8_t", "bc7Config[512][4]", [
        ((cfg["config"][r][0] | (128 if r in cfg["inferred"] else 0)), *cfg["config"][r][1:])
        if r >= 0 else (255, 0, 0, 0)
        for r in cfg["rank_of"]
    ])
    modes, plans = [], []
    for code in bc._BC6H_CODE[1:]:
        trans, base, deltas, regions = bc._BC6H_INFO[code]
        plan, _ = bc._bc6h_plan(code)
        modes.append((code, trans, base, deltas, regions, len(plans), len(plan)))
        plans.extend((c, q, bit) for c, q, bit, pos in plan)
    emit(out, "BC6Mode", "bc6Modes[14]", modes)
    emit(out, "int8_t", "bc6Plan[][3]", plans)


def write_profile(source, output):
    """Serialize local calibration JSON. Preserve both the available and confirmed precision."""
    ep, ix, ec, ic, ep_cb, ix_cb = bptc.load_bc6h(source, with_confirmed_bits=True)
    b7 = bptc.load_bc7(source)
    with Path(output).open("wb") as out:
        out.write(b"BCPHIST1")
        for i in range(256):
            out.write(struct.pack("<12H", *(v for channel in ep[i] for v in channel)))
            out.write(bytes(ix[i]))
            out.write(bytes(v for channel in ep_cb[i] for v in channel))
            out.write(bytes(ix_cb[i]))
            out.write(bytes((ec[i], ic[i])))
        for i in range(256):
            for key in ("history_endpoints", "history_endpoint_bits", "history_endpoint_confirmed_bits",
                        "history_indices", "history_index_bits", "history_index_confirmed_bits"):
                out.write(bytes(b7[key][i]))


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", help="the C++ tables to write (bcpack_tables.inc)")
    parser.add_argument("--profile", nargs=2, metavar=("JSON", "OUT"),
                        help="serialize a local calibration JSON to OUT instead of writing the tables")
    args = parser.parse_args()
    if args.profile:
        write_profile(*args.profile)
    elif args.output:
        with Path(args.output).open("w", encoding="utf-8", newline="\n") as output:
            generate(output)
    else:
        parser.error("name the tables file to write, or use --profile")
