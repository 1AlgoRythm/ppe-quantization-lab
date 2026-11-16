"""mAP50 / mAP50-95 of every ONNX variant on the held-out split, via ultralytics val.

ultralytics runs the .onnx files itself through ONNX Runtime (CPU EP), so the
numbers are for the actual exported/quantized graphs, not the .pt weights.
"""
import argparse, json
from pathlib import Path

import cv2
from ultralytics import YOLO


def n_images(name):
    """Images ultralytics keeps: it ignores images smaller than 10 px on a side."""
    return sum(min(cv2.imread(str(p)).shape[:2]) >= 10 for p in Path(f"data/{name}/test/images").glob("*.jpg"))


ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", default=["fp32", "fp16", "int8", "int8_naive"])
ap.add_argument("--out", default="results/accuracy.json")
a = ap.parse_args()

res = {}
for name in ("person", "ppe"):
    for v in a.variants:
        path = Path(f"models/{name}_{v}.onnx")
        if not path.exists():
            continue
        r = YOLO(str(path), task="detect").val(data=f"data/{name}.yaml", imgsz=640, batch=1, device="cpu",
                                               plots=False, verbose=False, project="data/runs", name="val", exist_ok=True)
        res[f"{name}_{v}"] = {"model": name, "variant": v, "map50": float(r.box.map50), "map50_95": float(r.box.map),
                              "precision": float(r.box.mp), "recall": float(r.box.mr),
                              "n_images": n_images(name), "evaluator": "ultralytics val on .onnx (ORT CPU EP), conf=0.001, iou=0.7"}
        print(name, v, res[f"{name}_{v}"]["map50"], res[f"{name}_{v}"]["map50_95"], flush=True)
Path(a.out).parent.mkdir(exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1))
