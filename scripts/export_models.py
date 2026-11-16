"""Export person.pt / ppe.pt to ONNX FP32 (ultralytics) and derive an FP16 ONNX.

FP16: ONNX Runtime's float16 converter, keep_io_types=True so the model still
takes/returns float32 (Cast nodes are inserted at the edges); weights and
internal ops are float16.
"""
import shutil
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from ultralytics import YOLO

for name in ("person", "ppe"):
    pt = Path("weights") / f"{name}.pt"
    exported = YOLO(str(pt)).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, batch=1)
    fp32 = Path("models") / f"{name}_fp32.onnx"
    shutil.move(exported, fp32)
    model = onnx.load(fp32)
    fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16, Path("models") / f"{name}_fp16.onnx")
    out = onnx.load(fp32).graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(name, "fp32 output shape:", dims)
