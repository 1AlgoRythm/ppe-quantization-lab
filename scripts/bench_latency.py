"""CPU latency, file size and peak memory for every ONNX variant.

Each variant runs in its OWN subprocess so peak RSS is not polluted by the
previous model. Timing is session.run() only (batch 1, 640x640 input),
ONNX Runtime CPU execution provider, 20 warmup + 200 timed runs.
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppe_lab.common import input_dtype, letterbox, make_session, to_tensor

WARMUP, RUNS = 20, 200


def sample_image(name):
    """A fixed real input: first test image (person) / largest test crop (ppe)."""
    paths = sorted(Path(f"data/{name}/test/images").glob("*.jpg"))
    if name == "ppe":
        paths = sorted(paths, key=lambda p: -cv2.imread(str(p)).size)
    return cv2.imread(str(paths[0]))


def child(name, variant):
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def watch():                                           # poll RSS every 2 ms
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)

    load0 = os.getloadavg()[0]                             # machine busy-ness, for honesty
    base = proc.memory_info().rss
    threading.Thread(target=watch, daemon=True).start()
    path = Path(f"models/{name}_{variant}.onnx")
    sess = make_session(path)
    x = to_tensor(letterbox(sample_image(name), 640)[0], input_dtype(sess))   # (1, 3, 640, 640)
    feed = {sess.get_inputs()[0].name: x}
    for _ in range(WARMUP):
        sess.run(None, feed)
    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t) * 1000)
    stop.set()
    t = np.array(times)
    print(json.dumps({"model": name, "variant": variant, "size_mb": path.stat().st_size / 1e6,
                      "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
                      "mean_ms": float(t.mean()), "min_ms": float(t.min()),
                      "peak_rss_mb": peak[0] / 1e6, "baseline_rss_mb": base / 1e6,
                      "warmup": WARMUP, "runs": RUNS, "input_shape": [1, 3, 640, 640],
                      "ort_threads": "default", "loadavg_before": load0, "loadavg_after": os.getloadavg()[0], "cpu_count": os.cpu_count()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("MODEL", "VARIANT"))
    ap.add_argument("--variants", nargs="+", default=["fp32", "fp16", "int8", "int8_naive"])
    ap.add_argument("--out", default="results/latency.json")
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        sys.exit()
    res = {}
    for name in ("person", "ppe"):
        for v in a.variants:
            r = subprocess.run([sys.executable, __file__, "--child", name, v], capture_output=True, text=True, check=True)
            res[f"{name}_{v}"] = json.loads(r.stdout.strip().splitlines()[-1])
            print(name, v, f"median {res[f'{name}_{v}']['median_ms']:.2f} ms", flush=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
