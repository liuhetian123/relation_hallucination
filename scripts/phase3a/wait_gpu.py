#!/usr/bin/env python3
"""Print the first idle GPU with enough free memory, optionally waiting."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def stats() -> list[tuple[int, int, int]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, free, util = [p.strip() for p in line.split(",")]
        rows.append((int(idx), int(free), int(util)))
    return rows


def pick(min_free_mb: int, max_util: int, n: int = 1) -> list[int] | None:
    cands = [
        (idx, free, util)
        for idx, free, util in stats()
        if free >= min_free_mb and util <= max_util
    ]
    cands.sort(key=lambda row: -row[1])
    if len(cands) < n:
        return None
    return [row[0] for row in cands[:n]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-free-mb", type=int, default=24000)
    p.add_argument("--max-util", type=int, default=8)
    p.add_argument("--poll-sec", type=int, default=60)
    p.add_argument("--timeout-sec", type=int, default=0, help="0 = wait forever")
    p.add_argument("--once", action="store_true")
    p.add_argument("--n", type=int, default=1, help="How many idle GPUs to return.")
    args = p.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    t0 = time.time()
    while True:
        rows = stats()
        gpus = pick(args.min_free_mb, args.max_util, args.n)
        msg = " ".join(f"gpu{i}:free={f}MiB,util={u}%" for i, f, u in rows)
        if gpus is not None:
            print(f"chosen_gpu={' '.join(map(str, gpus))}  {msg}", file=sys.stderr, flush=True)
            print(" ".join(map(str, gpus)))
            return
        print(f"waiting_gpu  {msg}", file=sys.stderr, flush=True)
        if args.once:
            sys.exit(1)
        if args.timeout_sec and (time.time() - t0) >= args.timeout_sec:
            sys.exit(2)
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
