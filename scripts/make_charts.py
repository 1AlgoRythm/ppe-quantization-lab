"""Charts for the README, drawn only from results/*.json.  usage: .venv/bin/python scripts/make_charts.py"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from chart_style import setup, bars, save, legend_top

ROOT = Path(__file__).resolve().parents[1]
acc = json.load(open(ROOT / "results/accuracy.json"))
lat = json.load(open(ROOT / "results/latency.json"))
cas = json.load(open(ROOT / "results/cascade.json"))
(ROOT / "docs").mkdir(exist_ok=True)
VARIANTS, MODELS = ["fp32", "fp16", "int8"], ["person", "ppe"]

fig, ax = setup(1, 3, w=12, h=4.2)
panels = [("mAP@0.5 on the held-out split", lambda m, v: acc[f"{m}_{v}"]["map50"], "{:.3f}"),
          ("Median latency, ms (batch 1, 640x640, CPU)", lambda m, v: lat[f"{m}_{v}"]["median_ms"], "{:.1f}"),
          ("File size, MB", lambda m, v: lat[f"{m}_{v}"]["size_mb"], "{:.1f}")]
for a, (t, get, fmt) in zip(ax, panels):
    bars(a, VARIANTS, MODELS, [[get(m, v) for v in VARIANTS] for m in MODELS], fmt)
    a.set_title(t, fontsize=10, color="#52514e", loc="left")
ax[0].set_ylim(0, 1.08); legend_top(fig, MODELS)
save(fig, ROOT / "docs/variants.png", "FP32 vs FP16 vs INT8, per model",
     "Source: results/accuracy.json, results/latency.json. The naive INT8 control (mAP 0) is left out; see the table. Latency measured on a shared, loaded machine.")

fig, ax = setup(w=7, h=4)
bars(ax, VARIANTS, ["end to end"], [[cas[v]["e2e_median_ms"] for v in VARIANTS]], "{:.1f} ms", width=0.5)
ax.set_ylabel("ms per image (median)")
save(fig, ROOT / "docs/cascade.png", "Two-stage cascade latency per image",
     "Source: results/cascade.json (person model, then the PPE model on each person crop; 40 images x 3 passes).")
print("wrote docs/variants.png, docs/cascade.png")
