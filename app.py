"""Streamlit app: compare model variants side by side + latency-vs-mAP chart.

Run:  streamlit run app.py
Needs models/*.onnx, data/ (scripts/prepare_data.py) and results/*.json.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ppe_lab.common import PERSON_NAMES, PPE_NAMES, detect, draw, make_session

VARIANTS = ["fp32", "fp16", "int8", "int8_naive"]


@st.cache_resource
def session(name, variant):
    return make_session(f"models/{name}_{variant}.onnx")


def run_variant(model, variant, img, conf):
    """Returns (annotated BGR image, n detections, milliseconds)."""
    t = time.perf_counter()
    if model == "cascade":
        person, ppe = session("person", variant), session("ppe", variant)
        persons = detect(person, img, conf=conf)
        out = draw(img, persons, PERSON_NAMES, (0, 200, 0))
        n = len(persons)
        for x1, y1, x2, y2, _, _ in persons:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            if ix2 - ix1 < 10 or iy2 - iy1 < 10:
                continue
            items = [(a + ix1, b + iy1, c + ix1, d + iy1, s, k)
                     for a, b, c, d, s, k in detect(ppe, img[iy1:iy2, ix1:ix2], conf=conf)]
            out = draw(out, items, PPE_NAMES, (0, 0, 255))
            n += len(items)
    else:
        names = PERSON_NAMES if model == "person" else PPE_NAMES
        dets = detect(session(model, variant), img, conf=conf)
        out, n = draw(img, dets, names), len(dets)
    return out, n, (time.perf_counter() - t) * 1000


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


st.title("PPE quantization lab")
model = st.sidebar.selectbox("Model", ["cascade", "person", "ppe"])
chosen = st.sidebar.multiselect("Variants", VARIANTS, default=["fp32", "int8"])
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
upload = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
tests = sorted(Path("data/person/test/images").glob("*.jpg"))
pick = st.sidebar.selectbox("...or a held-out test image", tests, format_func=lambda p: p.name[:40]) if tests else None

img = None
if upload is not None:
    img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
elif pick is not None:
    img = cv2.imread(str(pick))

if img is not None and chosen:
    cols = st.columns(len(chosen))
    for col, v in zip(cols, chosen):
        out, n, ms = run_variant(model, v, img, conf)
        col.image(out[:, :, ::-1], caption=f"{v}: {n} detections, {ms:.0f} ms (one run, includes pre/post)")
elif img is None:
    st.info("Run scripts/prepare_data.py first, or upload an image.")

st.header("Latency vs mAP50 (from results/*.json)")
acc, lat = load("results/accuracy.json"), load("results/latency.json")
rows = [{"model": k.split("_")[0], "variant": k.split("_", 1)[1], "mAP50": acc[k]["map50"],
         "median_ms": lat[k]["median_ms"], "size_mb": lat[k]["size_mb"]} for k in acc if k in lat]
if rows:
    df = pd.DataFrame(rows)
    show = st.selectbox("Chart model", ["person", "ppe"])
    d = df[df.model == show]
    st.scatter_chart(d, x="median_ms", y="mAP50", color="variant", size="size_mb")
    st.dataframe(d, hide_index=True)
else:
    st.warning("results/accuracy.json and results/latency.json not found. Run the eval and bench scripts.")
