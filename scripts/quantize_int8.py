"""INT8 static quantization (QDQ format) with ONNX Runtime.

Calibration images come from the TRAIN split only (data/<model>/train/images).
Activations are quantized to uint8, weights to int8, per-channel weights.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `ppe_lab` imports work

import cv2
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from ppe_lab.common import letterbox, to_tensor


class TrainImageReader(CalibrationDataReader):
    """Feeds letterboxed train images, one (1, 3, 640, 640) tensor at a time."""

    def __init__(self, image_dir, input_name, n):
        paths = sorted(Path(image_dir).glob("*.jpg"))
        step = max(1, len(paths) // n)                    # spread picks over the whole split
        self.paths = paths[::step][:n]
        self.input_name = input_name
        self.it = iter(self.paths)

    def get_next(self):
        p = next(self.it, None)
        if p is None:
            return None
        img = cv2.imread(str(p))
        if img is None or min(img.shape[:2]) < 10:
            return self.get_next()
        lb, _, _, _ = letterbox(img, 640)
        return {self.input_name: to_tensor(lb)}           # (1, 3, 640, 640)


def head_nodes_to_skip(model_path, mode):
    """Names of Detect-head nodes to leave in float.

    The head ends with Reshape/Concat/Sigmoid/DFL-softmax/box-decode ops (model.22/...).
    Box coordinates go up to ~640 and class scores live in 0..1, so squeezing them
    through one 8-bit scale destroys the scores. mode:
      none  quantize everything
      post  skip the decode ops after the last convs (Concat, Sigmoid, DFL, Slice, Mul ...)
      final post + the last 1x1 conv of every cv2/cv3 branch (the raw box/class logits)
    """
    import onnx
    graph = onnx.load(model_path).graph
    head = max({n.name.split("/")[1] for n in graph.node if n.name.startswith("/model.")},
               key=lambda m: int(m.split(".")[1]))
    skip = []
    for n in graph.node:
        if not n.name.startswith(f"/{head}/") or mode == "none":
            continue
        in_branch = "/cv2." in n.name or "/cv3." in n.name
        last_conv = in_branch and n.op_type == "Conv" and n.name.split("/")[-2].endswith(".2")
        if not in_branch or (mode == "final" and last_conv):
            skip.append(n.name)
    return skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--method", default="minmax", choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--models", nargs="+", default=["person", "ppe"])
    ap.add_argument("--suffix", default="int8")
    ap.add_argument("--head", default="post", choices=["none", "post", "final"])
    a = ap.parse_args()
    method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile}[a.method]
    for name in a.models:
        src, pre = Path(f"models/{name}_fp32.onnx"), Path(f"models/{name}_fp32_pre.onnx")
        quant_pre_process(str(src), str(pre))             # shape inference + fusions
        reader = TrainImageReader(f"data/{name}/train/images", "images", a.n_calib)
        dst = Path(f"models/{name}_{a.suffix}.onnx")
        quantize_static(str(pre), str(dst), reader, quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True, calibrate_method=method,
                        nodes_to_exclude=head_nodes_to_skip(str(pre), a.head))
        pre.unlink()
        print(f"{dst}: {dst.stat().st_size / 1e6:.2f} MB, calibrated on {len(reader.paths)} train images, method={a.method}, head={a.head}")


if __name__ == "__main__":
    main()
