"""Tests for the built-in, dependency-free sixel encoder.

A minimal in-test sixel *decoder* lets us assert exact round-trips without
needing chafa/img2sixel/ImageMagick on PATH.
"""

import re

import numpy as np

import plotty


# ---- a tiny sixel decoder (only the subset plotty emits) --------------------

def _decode_sixel(data):
    assert data[:3] == b"\x1bPq"
    assert data[-2:] == b"\x1b\\"
    s = data[3:-2]
    pos, n = 0, len(s)

    assert s[pos:pos + 1] == b'"'                # raster attributes
    pos += 1
    m = re.match(rb"(\d+);(\d+);(\d+);(\d+)", s[pos:])
    _, _, w, h = (int(x) for x in m.groups())
    pos += m.end()

    palette = {}
    img = np.zeros((h, w, 3), np.uint8)
    band = np.full((6, w), -1, np.int32)
    top, x, color = 0, 0, 0

    def commit():
        for r in range(6):
            row = top + r
            if row >= h:
                continue
            for c in np.nonzero(band[r] >= 0)[0]:
                img[row, c] = palette[int(band[r, c])]

    def put(val):
        nonlocal x
        if x < w:
            bits = val - 63
            for r in range(6):
                if bits >> r & 1:
                    band[r, x] = color
        x += 1

    while pos < n:
        ch = s[pos:pos + 1]
        if ch == b"#":
            pos += 1
            m = re.match(rb"(\d+)", s[pos:])
            idx = int(m.group(1))
            pos += m.end()
            if s[pos:pos + 1] == b";":           # palette definition
                m = re.match(rb";2;(\d+);(\d+);(\d+)", s[pos:])
                r100, g100, b100 = (int(v) for v in m.groups())
                pos += m.end()
                palette[idx] = (round(r100 * 255 / 100),
                                round(g100 * 255 / 100),
                                round(b100 * 255 / 100))
            else:                                # colour selection
                color = idx
        elif ch == b"!":                         # run-length
            pos += 1
            m = re.match(rb"(\d+)", s[pos:])
            run = int(m.group(1))
            pos += m.end()
            val = s[pos]
            pos += 1
            for _ in range(run):
                put(val)
        elif ch == b"$":                         # carriage return
            pos += 1
            x = 0
        elif ch == b"-":                         # next band
            pos += 1
            commit()
            band[:] = -1
            top += 6
            x = 0
        else:
            put(s[pos])
            pos += 1
    commit()
    return img


# ---- _rle -------------------------------------------------------------------

def test_rle_short_runs_inline():
    assert plotty._rle(np.array([65, 65, 66], np.int64)) == b"AAB"


def test_rle_long_run_compressed():
    assert plotty._rle(np.array([63, 63, 63, 63], np.int64)) == b"!4?"


# ---- _quantize --------------------------------------------------------------

def test_quantize_caps_at_256_colors():
    rng = np.arange(300 * 3, dtype=np.uint8).reshape(300, 3)  # 300 distinct rows
    img = np.tile(rng[:, None, :], (1, 4, 1))                 # 300x4x3
    pal, idx = plotty._quantize(img)
    assert pal.shape[0] <= 256
    assert idx.min() >= 0 and idx.max() < pal.shape[0]
    assert idx.shape[0] == img.shape[0] * img.shape[1]


def test_quantize_single_color():
    img = np.full((5, 5, 3), 17, np.uint8)
    pal, idx = plotty._quantize(img)
    assert pal.shape[0] == 1
    assert set(idx.tolist()) == {0}


def test_quantize_fast_path_exact_for_many_distinct_colors():
    # 200 distinct colors (<= 256): exact palette, perfect reconstruction
    rows = np.arange(200, dtype=np.uint8)
    img = np.stack([rows, rows[::-1], np.full(200, 7, np.uint8)], axis=-1)
    img = img[None].repeat(3, axis=0)            # (3, 200, 3)
    pal, idx = plotty._quantize(img)
    assert len(np.unique(pal, axis=0)) == 200
    assert np.array_equal(pal[idx].reshape(img.shape), img)


# ---- background compositing ---------------------------------------------------

def test_parse_bg():
    assert plotty._parse_bg("#1e1e2e") == (0x1E, 0x1E, 0x2E)
    assert plotty._parse_bg("ff0000") == (255, 0, 0)
    assert plotty._parse_bg(None) == (255, 255, 255)


def test_parse_bg_invalid_warns_and_falls_back(capsys):
    assert plotty._parse_bg("not-a-color") == (255, 255, 255)
    assert "invalid bg" in capsys.readouterr().err


def test_load_rgb_composites_alpha_over_bg(tmp_path):
    import matplotlib.image as mpimg
    rgba = np.zeros((4, 4, 4), np.uint8)          # fully transparent image
    rgba[:2, :2] = (10, 20, 30, 255)              # one opaque corner
    png = tmp_path / "alpha.png"
    mpimg.imsave(str(png), rgba)
    plotty._cfg["bg"] = "#ff0000"
    img = plotty._load_rgb(str(png))
    assert tuple(img[3, 3]) == (255, 0, 0)        # transparent -> bg color
    assert tuple(img[0, 0]) == (10, 20, 30)       # opaque untouched


# ---- _resize ----------------------------------------------------------------

def test_resize_shapes():
    img = np.zeros((8, 6, 3), np.uint8)
    assert plotty._resize(img, 3, 4).shape == (4, 3, 3)
    assert plotty._resize(img, 6, 8) is not None
    assert plotty._resize(img, 6, 8).shape == (8, 6, 3)  # identity branch


# ---- _target_size -----------------------------------------------------------

def test_target_size_scales_to_width(monkeypatch):
    # cols=80, rows=24, 800x480 px -> 10x20 px cells; size=60 -> target 600px wide
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (80, 24, 800, 480))
    plotty._cfg["size"] = 60
    assert plotty._target_size(0, 1200, 600) == (600, 300)   # downscaled to width
    assert plotty._target_size(0, 450, 300) == (600, 400)    # upscaled to width


def test_target_size_pixel_fallback(monkeypatch):
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (80, 24, 0, 0))
    plotty._cfg["size"] = 40
    tw, th = plotty._target_size(0, 800, 400)
    assert tw == 400 and th == 200                           # 40 cells * 10px default


def test_target_size_height_bound(monkeypatch):
    # a very tall image: pane height limits the scale, not width
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (80, 24, 800, 480))
    plotty._cfg["size"] = 80
    tw, th = plotty._target_size(0, 400, 4000)
    assert th == 460 and tw == round(400 * 460 / 4000)       # max_h = (24-1)*20


# ---- full encode/decode round-trip -----------------------------------------

def test_encode_decode_roundtrip_exact():
    h, w = 13, 10                                            # height not a multiple of 6
    img = np.zeros((h, w, 3), np.uint8)
    img[:6] = (255, 0, 0)
    img[6:] = (0, 0, 255)
    img[:, 3] = (0, 153, 0)
    img[2, :] = (102, 102, 102)
    pal, idx = plotty._quantize(img)
    data = plotty._sixel_bytes(pal, idx, h, w)

    assert data.startswith(b"\x1bPq")
    assert data.endswith(b"\x1b\\")
    assert b'"1;1;%d;%d' % (w, h) in data

    decoded = _decode_sixel(data)

    def rt(c):  # mirror the 0-100 colour resolution sixel imposes
        return round(round(int(c) * 100 / 255) * 255 / 100)

    expected = np.array([[rt(v) for v in col] for col in pal], np.uint8)[idx]
    assert np.array_equal(decoded, expected.reshape(h, w, 3))


# ---- _resolve_cmd (renderer size placeholders) ------------------------------

def test_resolve_cmd_substitution(monkeypatch):
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (80, 24, 0, 0))
    plotty._cfg["size"] = 48
    assert plotty._resolve_cmd("chafa --size {size}", 1) == "chafa --size 48"
    assert plotty._resolve_cmd("img2sixel -w {width}", 1) == "img2sixel -w 480"
    assert plotty._resolve_cmd("img2sixel", 1) == "img2sixel"
    assert plotty._resolve_cmd(None, 1) is None


# ---- _render_bytes dispatch -------------------------------------------------

def test_render_bytes_external_captures_stdout(fake_run):
    fake_run.default = b"EXTERNAL-SIXEL"
    plotty._cfg["imgcat"] = "img2sixel"
    out = plotty._render_bytes("/tmp/whatever.png", 1)
    assert out == b"EXTERNAL-SIXEL"
    assert fake_run.calls[0][:2] == ["sh", "-c"]
    assert "img2sixel" in fake_run.calls[0][2]


def test_render_bytes_builtin_scales_with_size(tmp_path):
    import matplotlib.image as mpimg
    arr = np.zeros((100, 400, 3), np.uint8)
    arr[:, :200] = (220, 40, 40)
    png = tmp_path / "fig.png"
    mpimg.imsave(str(png), arr)
    plotty._cfg["imgcat"] = None

    plotty._cfg["size"] = 20
    small = plotty._render_bytes(str(png), 1)
    plotty._cfg["size"] = 60
    big = plotty._render_bytes(str(png), 1)

    for data in (small, big):
        assert data.startswith(b"\x1bPq") and data.endswith(b"\x1b\\")
    assert len(small) < len(big)                 # larger size -> more sixel data
