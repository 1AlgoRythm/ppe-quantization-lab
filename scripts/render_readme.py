"""Fill the results tables in README.md from results/*.json (nothing is typed by hand)."""
import json
import re
from pathlib import Path

VARIANTS = ["fp32", "fp16", "int8", "int8_naive"]


def load(p):
    f = Path(p)
    return json.loads(f.read_text()) if f.exists() else {}


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    acc, lat, cas = load("results/accuracy.json"), load("results/latency.json"), load("results/cascade.json")
    parts = []
    for name in ("person", "ppe"):
        rows = []
        for v in VARIANTS:
            k = f"{name}_{v}"
            if k not in acc or k not in lat:
                continue
            a, l = acc[k], lat[k]
            rows.append([v, f"{a['map50']:.4f}", f"{a['map50_95']:.4f}", f"{l['size_mb']:.2f}", f"{l['median_ms']:.2f}",
                         f"{l['p95_ms']:.2f}", f"{l['peak_rss_mb']:.0f}"])
        n = acc.get(f"{name}_fp32", {}).get("n_images", "?")
        parts.append(f"**{name}.pt** ({n} held-out {'images' if name == 'person' else 'person crops'})\n\n" + table(
            ["variant", "mAP50", "mAP50-95", "size MB", "median ms", "p95 ms", "peak RSS MB"], rows))
    rows = [[v, c["e2e_median_ms"] and f"{c['e2e_median_ms']:.1f}", f"{c['e2e_p95_ms']:.1f}",
             f"{c['person_stage_median_ms']:.1f}", f"{c['ppe_stage_median_ms']:.1f}",
             f"{c['persons_per_image_mean']:.2f}", f"{c['ppe_dets_per_image_mean']:.2f}"] for v, c in cas.items()]
    parts.append("**Two-stage cascade, end to end per image** (40 held-out images x 3 timed passes)\n\n" + table(
        ["variant", "median ms", "p95 ms", "person stage median ms", "PPE stage median ms",
         "persons/image", "PPE boxes/image"], rows))
    block = "\n\n".join(parts)
    readme = Path("README.md").read_text()
    new = re.sub(r"(<!-- results:start -->).*?(<!-- results:end -->)", lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}",
                 readme, flags=re.S)
    Path("README.md").write_text(new)
    print(block)


if __name__ == "__main__":
    main()
