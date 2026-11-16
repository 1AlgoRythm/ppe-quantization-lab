"""Shared helpers: letterbox, ONNX Runtime session, YOLOv8 decode + NMS.

Everything here works on plain numpy arrays. Tensor shapes are in comments.
"""
import cv2
import numpy as np
import onnxruntime as ort

PERSON_NAMES = ["person"]
# ppe.pt only has 7 classes (checked with YOLO("weights/ppe.pt").names).
PPE_NAMES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit"]


def letterbox(img, size=640):
    """Resize keeping aspect ratio, pad to size x size with grey (114).

    Returns the padded image plus (scale, pad_x, pad_y) to undo it later.
    """
    h, w = img.shape[:2]                                   # (H, W, 3) BGR
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    bottom, right = size - nh - top, size - nw - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, scale, left, top                           # (size, size, 3)


def to_tensor(img_letterboxed, dtype=np.float32):
    """BGR HWC uint8 -> RGB NCHW float in [0, 1]."""
    rgb = img_letterboxed[:, :, ::-1]                      # (640, 640, 3) RGB
    chw = np.transpose(rgb, (2, 0, 1))                     # (3, 640, 640)
    return np.ascontiguousarray(chw[None], dtype=dtype) / dtype(255)  # (1, 3, 640, 640)


def make_session(path, threads=0):
    """ONNX Runtime session on the CPU execution provider only."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads                    # 0 = ORT default (all cores)
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def input_dtype(sess):
    return np.float16 if "float16" in sess.get_inputs()[0].type else np.float32


def decode(output, scale, pad_x, pad_y, orig_w, orig_h, conf=0.25, iou=0.45):
    """YOLOv8 head output -> list of (x1, y1, x2, y2, score, class_id).

    output: (1, 4 + nc, 8400); rows 0..3 are cx, cy, w, h in letterboxed pixels,
    the remaining nc rows are per-class scores (already sigmoid-ed).
    """
    pred = output[0].astype(np.float32).T                  # (8400, 4 + nc)
    scores_all = pred[:, 4:]                               # (8400, nc)
    cls = scores_all.argmax(axis=1)                        # (8400,)
    score = scores_all[np.arange(len(cls)), cls]           # (8400,)
    keep = score >= conf
    pred, cls, score = pred[keep], cls[keep], score[keep]
    if len(score) == 0:
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (cx - w / 2 - pad_x) / scale                      # undo letterbox
    y1 = (cy - h / 2 - pad_y) / scale
    boxes = np.stack([x1, y1, w / scale, h / scale], 1)    # (N, 4) x, y, w, h
    idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), score.tolist(), cls.tolist(), conf, iou)
    dets = []
    for i in np.array(idx).reshape(-1):
        bx, by, bw, bh = boxes[i]
        x1c, y1c = max(0.0, bx), max(0.0, by)
        x2c, y2c = min(float(orig_w), bx + bw), min(float(orig_h), by + bh)
        dets.append((x1c, y1c, x2c, y2c, float(score[i]), int(cls[i])))
    return dets


def detect(sess, img, size=640, conf=0.25, iou=0.45):
    """Full single-image detect: letterbox -> run -> decode."""
    h, w = img.shape[:2]
    lb, scale, px, py = letterbox(img, size)
    x = to_tensor(lb, input_dtype(sess))                   # (1, 3, 640, 640)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]  # (1, 4 + nc, 8400)
    return decode(out, scale, px, py, w, h, conf, iou)


def draw(img, dets, names, color=(0, 200, 0)):
    """Draw boxes + 'name score' labels on a copy of img."""
    out = img.copy()
    for x1, y1, x2, y2, s, k in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, f"{names[k]} {s:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
