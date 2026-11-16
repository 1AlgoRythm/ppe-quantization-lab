"""End-to-end two-stage cascade per variant: person detect -> crop -> PPE detect.

Runs on the 40 held-out whole images. Both models use the same variant.
Time per image = letterbox + inference + decode for the person model, plus the
same three steps for every person crop. One untimed warmup pass, then 3 timed passes.
"""
import argparse, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppe_lab.common import detect, make_session


def cascade(person, ppe, img, conf=0.25):
    """Returns (persons, ppe_dets_in_full_image_coords, t_person_ms, t_ppe_ms)."""
    t0 = time.perf_counter()
    persons = detect(person, img, conf=conf)
    t1 = time.perf_counter()
    items = []
    for x1, y1, x2, y2, _, _ in persons:
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        if ix2 - ix1 < 10 or iy2 - iy1 < 10:
            continue
        crop = img[iy1:iy2, ix1:ix2]                       # (h, w, 3)
        for a, b, c, d, s, k in detect(ppe, crop, conf=conf):
            items.append((a + ix1, b + iy1, c + ix1, d + iy1, s, k))
    t2 = time.perf_counter()
    return persons, items, (t1 - t0) * 1000, (t2 - t1) * 1000


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["fp32", "fp16", "int8", "int8_naive"])
    ap.add_argument("--out", default="results/cascade.json")
    a = ap.parse_args()
    paths = sorted(Path("data/person/test/images").glob("*.jpg"))
    imgs = [cv2.imread(str(p)) for p in paths]
    res = {}
    for v in a.variants:
        person, ppe = make_session(f"models/person_{v}.onnx"), make_session(f"models/ppe_{v}.onnx")
        load0 = os.getloadavg()[0]
        for img in imgs:                                   # warmup pass
            cascade(person, ppe, img)
        tot, tp, tq, n_p, n_i = [], [], [], [], []
        for _ in range(3):
            for img in imgs:
                t = time.perf_counter()
                persons, items, a1, a2 = cascade(person, ppe, img)
                tot.append((time.perf_counter() - t) * 1000)
                tp.append(a1); tq.append(a2); n_p.append(len(persons)); n_i.append(len(items))
        res[v] = {"variant": v, "n_images": len(imgs), "timed_passes": 3, "n_timings": len(tot),
                  "e2e_median_ms": float(np.median(tot)), "e2e_p95_ms": float(np.percentile(tot, 95)),
                  "person_stage_median_ms": float(np.median(tp)), "ppe_stage_median_ms": float(np.median(tq)),
                  "persons_per_image_mean": float(np.mean(n_p)), "ppe_dets_per_image_mean": float(np.mean(n_i)),
                  "conf": 0.25, "iou": 0.45, "loadavg_before": load0, "loadavg_after": os.getloadavg()[0]}
        print(v, f"e2e median {res[v]['e2e_median_ms']:.1f} ms", flush=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
