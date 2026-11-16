"""Unzip the dataset, make a seeded 70/20/10 split, and build the PPE crop sets.

Why: the original repo's test split was never saved (Split_dataset.py shuffles
without a seed and the zip has no train/val/test folders), so we regenerate a
deterministic split with the same ratios and logic, seed 0.

Outputs under data/:
  person/{train,val,test}/{images,labels}   whole images, class 0 = person
  ppe/{train,val,test}/{images,labels}      person crops (crop.py logic)
  person.yaml, ppe.yaml, split.json
"""
import argparse, json, random, shutil, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

N_PPE = 7  # ppe.pt knows classes 0..6; dataset classes 8, 9 (ear-protector, safety-harness) are dropped


def split_names(names, seed=0, ratios=(0.7, 0.2, 0.1)):
    """Same cutoffs as Split_dataset.py, but sorted + seeded so it is repeatable."""
    names = sorted(names)
    random.Random(seed).shuffle(names)
    a = int(ratios[0] * len(names))
    b = int((ratios[0] + ratios[1]) * len(names))
    return {"train": names[:a], "val": names[a:b], "test": names[b:]}


def crop_persons(xml_path, img, classes):
    """crop.py logic: one crop per 'person' box; PPE boxes re-expressed in crop coords.

    Yields (crop_index, crop_image (h, w, 3), yolo_lines, n_dropped_boxes). The dataset class index
    is shifted by -1 (person removed), and boxes of classes the model lacks are dropped.
    """
    root = ET.parse(xml_path).getroot()
    objs = []
    for o in root.findall("object"):
        b = o.find("bndbox")
        objs.append((o.find("name").text, [int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]))
    persons = [bb for n, bb in objs if n == "person"]
    H, W = img.shape[:2]
    for i, (x0, y0, x1, y1) in enumerate(persons):
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        cw, ch = x1 - x0, y1 - y0
        lines, dropped = [], 0
        for n, (ox0, oy0, ox1, oy1) in objs:
            if n == "person" or n not in classes:
                continue
            cid = classes.index(n) - 1
            if cid >= N_PPE:
                dropped += 1
                continue
            nx0, ny0 = max(0, ox0 - x0), max(0, oy0 - y0)
            nx1, ny1 = min(cw, ox1 - x0), min(ch, oy1 - y0)
            if nx1 <= nx0 or ny1 <= ny0:
                continue
            lines.append(f"{cid} {(nx0 + nx1) / 2 / cw:.6f} {(ny0 + ny1) / 2 / ch:.6f} "
                         f"{(nx1 - nx0) / cw:.6f} {(ny1 - ny0) / ch:.6f}")
        yield i, img[y0:y1, x0:x1], lines, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="path to datasets.zip")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    raw = out / "raw"
    if not raw.exists():
        zipfile.ZipFile(a.zip).extractall(raw)
    ds = raw / "datasets"
    classes = (ds / "classes.txt").read_text().split()
    stems = [p.stem for p in (ds / "images").glob("*.jpg") if (ds / "labels" / f"{p.stem}.txt").exists()]
    split = split_names(stems, a.seed)

    info = {"seed": a.seed, "classes_txt": classes, "n_images": len(stems), "splits": {}}
    for sp, names in split.items():
        n_crops = n_crops_with_labels = n_boxes = n_dropped = 0
        for kind in ("person", "ppe"):
            for sub in ("images", "labels"):
                (out / kind / sp / sub).mkdir(parents=True, exist_ok=True)
        for s in names:
            shutil.copy(ds / "images" / f"{s}.jpg", out / "person" / sp / "images" / f"{s}.jpg")
            shutil.copy(ds / "labels" / f"{s}.txt", out / "person" / sp / "labels" / f"{s}.txt")
            xml = ds / "annotations" / f"{s}.xml"
            img = cv2.imread(str(ds / "images" / f"{s}.jpg"))
            if not xml.exists() or img is None:
                continue
            for i, crop, lines, dropped in crop_persons(xml, img, classes):
                cs = f"{s}_person_{i}"
                cv2.imwrite(str(out / "ppe" / sp / "images" / f"{cs}.jpg"), crop)
                n_crops += 1
                n_dropped += dropped
                if lines:  # crops with no PPE boxes get no label file = background (as crop.py)
                    (out / "ppe" / sp / "labels" / f"{cs}.txt").write_text("\n".join(lines))
                    n_crops_with_labels += 1
                    n_boxes += len(lines)
        info["splits"][sp] = {"images": len(names), "ppe_crops": n_crops,
                              "ppe_crops_with_labels": n_crops_with_labels, "ppe_boxes": n_boxes, "dropped_boxes_unknown_class": n_dropped,
                              "files": names}
    root = out.resolve()
    (root / "person.yaml").write_text(
        f"path: {root}/person\ntrain: train/images\nval: test/images\ntest: test/images\nnames:\n  0: person\n")
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes[1:1 + N_PPE]))
    (root / "ppe.yaml").write_text(
        f"path: {root}/ppe\ntrain: train/images\nval: test/images\ntest: test/images\nnames:\n{names_yaml}\n")
    (out / "split.json").write_text(json.dumps(info, indent=1))
    for sp, d in info["splits"].items():
        print(sp, {k: v for k, v in d.items() if k != "files"})


if __name__ == "__main__":
    main()
