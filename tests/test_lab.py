import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ppe_lab.common import decode, letterbox, to_tensor
from prepare_data import split_names


def test_letterbox_shape_and_scale():
    img = np.zeros((300, 600, 3), np.uint8)
    out, scale, px, py = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert scale == 640 / 600 and px == 0 and py > 0
    assert to_tensor(out).shape == (1, 3, 640, 640)


def test_decode_recovers_box():
    # one anchor: box centre (320, 320), 100x200 in letterboxed pixels, class 1 with score 0.9
    out = np.zeros((1, 4 + 3, 8400), np.float32)
    out[0, :4, 5] = [320, 320, 100, 200]
    out[0, 4 + 1, 5] = 0.9
    dets = decode(out, scale=1.0, pad_x=0, pad_y=0, orig_w=640, orig_h=640)
    assert len(dets) == 1
    x1, y1, x2, y2, s, k = dets[0]
    assert (round(x1), round(y1), round(x2), round(y2), k) == (270, 220, 370, 420, 1)


def test_split_is_deterministic_and_disjoint():
    names = [f"img{i}" for i in range(395)]
    a, b = split_names(names, seed=0), split_names(list(reversed(names)), seed=0)
    assert a == b
    assert [len(a[k]) for k in ("train", "val", "test")] == [276, 79, 40]
    assert not set(a["train"]) & set(a["test"])


def test_streamlit_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
