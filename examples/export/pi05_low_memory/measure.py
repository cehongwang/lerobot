"""Run one command and record its peak host RSS and peak GPU memory, the same way for
every runner (Python or C++), so memory numbers are comparable across runtimes.

    python measure.py --gpu 0 --tag cpp_et -- ./et_bench --model=... [args]

Host: ru_maxrss of the child from wait4 (the kernel's own peak-RSS accounting).
GPU: NVML device-wide used memory sampled every 10 ms, minus the idle level measured
just before launch. NVML cannot attribute memory to a PID inside this container, so the
GPU must be otherwise idle while this runs. Writes RESULTS/<tag>.mem.json.

--timeline also writes RESULTS/<tag>.timeline.csv, one row every 0.2 s: host RSS of the child's
process tree, system-wide used memory (MemTotal - MemAvailable - Shmem, what a unified-memory Jetson
competes for) and GPU used memory above idle; and RESULTS/<tag>.log, the child's output with each
line prefixed by its elapsed time, so peaks can be matched to the phase that printed nearby.

--gpu -1 is for a Jetson, where NVML reports no memory. Used memory is MemTotal - MemAvailable - Shmem
(files written to tmpfs don't count) plus the hugepages in use: on a Thor configured with a hugepage pool,
CUDA allocations come from that pool and never touch MemAvailable (a 4 GiB cudaMalloc moves it by 20 MiB
and HugePages_Free by 4 GiB); on a Jetson without one the term is zero. The result is the peak of that
sum above its idle level, its hugepage part reported as gpu_peak_mb, plus the peak summed RSS of the
process tree. Both counters are system-wide, so other users' activity on the device shows up in them.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True, help="physical GPU index (NVML ignores CUDA_VISIBLE_DEVICES)")
    p.add_argument("--tag", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--timeline", action="store_true", help="also write <tag>.timeline.csv and <tag>.log")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)

    def meminfo():
        return {line.split(":")[0]: int(line.split()[1]) * 1024 for line in open("/proc/meminfo")}

    def hugepages_used(m):
        return (m["HugePages_Total"] // 1024 - m["HugePages_Free"] // 1024) * m["Hugepagesize"]

    unified = a.gpu < 0
    if unified:
        def gpu_used():
            return hugepages_used(meminfo())
    else:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(a.gpu)

        def gpu_used():
            return pynvml.nvmlDeviceGetMemoryInfo(h).used

    def sys_used():
        m = meminfo()
        return m["MemTotal"] - m["MemAvailable"] - m["Shmem"] + hugepages_used(m)

    idle = gpu_used()
    sys_idle = sys_used()
    peak = [idle]
    sys_peak = [sys_idle]
    tree_peak = [0]
    done = threading.Event()
    env = dict(os.environ, PYTHONUNBUFFERED="1", **({} if unified else {"CUDA_VISIBLE_DEVICES": str(a.gpu)}))
    t = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, **({"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
                                             if a.timeline else {}))

    def tree_rss():
        try:
            root = psutil.Process(proc.pid)
            procs = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return 0
        total = 0
        for q in procs:
            try:
                total += q.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total

    def poll():
        rows = open(res / f"{a.tag}.timeline.csv", "w") if a.timeline else None  # noqa: SIM115
        if rows:
            rows.write("t_s,host_rss_mb,sys_used_mb,gpu_mb\n")
        next_row = 0.0
        while not done.is_set():
            used = gpu_used()
            peak[0] = max(peak[0], used)
            now = time.perf_counter() - t
            if unified or (rows and now >= next_row):
                rss, sysu = tree_rss(), sys_used()
                tree_peak[0], sys_peak[0] = max(tree_peak[0], rss), max(sys_peak[0], sysu)
            if rows and now >= next_row:
                rows.write(f"{now:.2f},{rss / 2**20:.0f},{sysu / 2**20:.0f},{(used - idle) / 2**20:.0f}\n")
                rows.flush()
                next_row = now + 0.2
            # ponytail: walking the process tree every 50 ms costs a little CPU; NVML alone polls at 10 ms.
            time.sleep(0.05 if unified else 0.01)
        if rows:
            rows.close()

    def tee():
        with open(res / f"{a.tag}.log", "w") as log:
            for raw in proc.stdout:
                line = raw.decode(errors="replace")
                sys.stdout.write(line)
                log.write(f"{time.perf_counter() - t:9.2f}  {line}")

    threads = [threading.Thread(target=poll, daemon=True)]
    if a.timeline:
        threads.append(threading.Thread(target=tee, daemon=True))
    for th in threads:
        th.start()
    _, status, ru = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t
    done.set()
    for th in threads:
        th.join()
    r = {"tag": a.tag, "cmd": cmd, "exit": os.waitstatus_to_exitcode(status), "wall_s": wall,
         "host_peak_rss_mb": ru.ru_maxrss / 1024, "gpu_peak_mb": (peak[0] - idle) / 2**20,
         "gpu_idle_mb": idle / 2**20, "tree_peak_rss_mb": tree_peak[0] / 2**20,
         "sys_peak_above_idle_mb": (sys_peak[0] - sys_idle) / 2**20, "sys_idle_mb": sys_idle / 2**20}
    (res / f"{a.tag}.mem.json").write_text(json.dumps(r, indent=2))
    print(f"[measure] {json.dumps(r)}", flush=True)
    raise SystemExit(r["exit"])


if __name__ == "__main__":
    main()
