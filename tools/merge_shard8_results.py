import argparse
import math
import re
from pathlib import Path

SUBJECT_RANGES = {
    0: (0, 2),
    1: (2, 4),
    2: (4, 6),
    3: (6, 8),
    4: (8, 10),
    5: (10, 12),
    6: (12, 14),
    7: (14, 15),
}


def parse_each_acc(log_text: str):
    m = re.findall(r"final each acc:\s*([0-9eE\.\,\-\+ ]+)", log_text)
    if not m:
        raise ValueError("missing 'final each acc:' in log")
    last = m[-1]
    vals = [float(x.strip()) for x in last.split(",") if x.strip()]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, required=True, help="e.g. baseline_seed3_s1_shard8_*.log")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    all_logs = sorted(log_dir.glob(args.pattern))
    if not all_logs:
        raise FileNotFoundError(f"no logs matched: {log_dir / args.pattern}")

    merged = [None] * 15
    for p in all_logs:
        m = re.search(r"shard8_(\d+)", p.name)
        if not m:
            continue
        shard = int(m.group(1))
        if shard not in SUBJECT_RANGES:
            continue
        values = parse_each_acc(p.read_text(encoding="utf-8", errors="ignore"))
        s, e = SUBJECT_RANGES[shard]
        expected = e - s
        if len(values) != expected:
            raise ValueError(f"{p.name}: expected {expected} values, got {len(values)}")
        for i, v in enumerate(values):
            merged[s + i] = v

    if any(v is None for v in merged):
        missing = [i for i, v in enumerate(merged) if v is None]
        raise RuntimeError(f"missing subject acc for indices: {missing}")

    avg = sum(merged) / len(merged)
    std = math.sqrt(sum((x - avg) ** 2 for x in merged) / len(merged))
    print("final each acc:", ",".join(f"{x:.6f}" for x in merged))
    print(f"final acc avg: {avg:.6f}")
    print(f"final acc std: {std:.6f}")


if __name__ == "__main__":
    main()

