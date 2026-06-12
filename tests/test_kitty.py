"""Tests for the kitty-graphics encoder (Unicode placeholders, imgcat="kitty").

All structural: APC framing, chunking, base64 round-trip, placeholder grid,
and tmux passthrough wrapping — no kitty/ghostty needed.
"""

import io
import os
import re
import base64

import numpy as np

import plotty

PLACEHOLDER = "\U0010EEEE"
APC = re.compile(rb"\x1b_G([^;\x1b]*)(?:;([^\x1b]*))?\x1b\\")


def _png(tmp_path, w=40, h=20, alpha=False):
    import matplotlib.image as mpimg
    if alpha:
        arr = np.zeros((h, w, 4), np.uint8)
        arr[: h // 2, :] = (10, 20, 30, 255)      # top opaque, bottom transparent
    else:
        arr = np.zeros((h, w, 3), np.uint8)
        arr[:, : w // 2] = (200, 30, 30)
    p = tmp_path / "img.png"
    mpimg.imsave(str(p), arr)
    return p


def test_kitty_stream_structure_and_roundtrip(monkeypatch, tmp_path):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (100, 50, 1000, 1000))
    plotty._cfg.update(imgcat="kitty", size=40, bg=None)
    png_path = _png(tmp_path)
    out = plotty._render_bytes(str(png_path), 1)

    apcs = APC.findall(out)
    assert len(apcs) >= 2
    assert apcs[0][0].decode().startswith("a=d,d=I")    # old image freed first
    head = apcs[1][0].decode()
    for key in ("a=T", "U=1", "q=2", "f=100", "t=d", "c=", "r=", f"i={plotty._kitty_id()}"):
        assert key in head, f"missing {key} in {head}"
    assert "m=0" in apcs[-1][0].decode()                # stream closed
    # base64 payload reassembles to the exact PNG on disk
    data = b"".join(p for _, p in apcs[1:])
    assert base64.standard_b64decode(data) == png_path.read_bytes()


def test_kitty_chunking_large_png(monkeypatch, tmp_path):
    import matplotlib.image as mpimg
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (100, 50, 0, 0))
    noise = (np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8)
    p = tmp_path / "big.png"
    mpimg.imsave(str(p), noise)
    plotty._cfg.update(imgcat="kitty", size=40, bg=None)
    out = plotty._render_bytes(str(p), 1)

    apcs = APC.findall(out)
    data_chunks = [d for _, d in apcs[1:]]
    assert len(data_chunks) >= 2                        # actually chunked
    assert all(len(d) == 4096 for d in data_chunks[:-1])  # full + 4-byte aligned
    assert len(data_chunks[-1]) <= 4096
    assert "m=1" in apcs[1][0].decode()
    assert "m=0" in apcs[-1][0].decode()
    assert all(k.decode() == "m=1" for k, _ in apcs[2:-1])  # continuations: only m
    assert base64.standard_b64decode(b"".join(data_chunks)) == p.read_bytes()


def test_kitty_placeholder_grid(monkeypatch, tmp_path):
    monkeypatch.delenv("TMUX", raising=False)
    # 100x50 cells, 1000x1000 px -> 10x20 px cells; 40x20 px image, size=20
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (100, 50, 1000, 1000))
    plotty._cfg.update(imgcat="kitty", size=20, bg=None)
    out = plotty._render_bytes(str(_png(tmp_path, w=40, h=20)), 1).decode("utf-8")

    assert f"\x1b[38;5;{plotty._kitty_id()}m" in out    # image id in fg color
    rows = [ln for ln in out.split("\r\n") if PLACEHOLDER in ln]
    # r = round(c * cell_w * ih / (iw * cell_h)) = round(20*10*20/(40*20)) = 5
    assert len(rows) == 5
    assert all(ln.count(PLACEHOLDER) == 20 for ln in rows)
    d = plotty._DIACRITICS
    # explicit (row, col) diacritics on every cell
    assert PLACEHOLDER + chr(d[0]) + chr(d[0]) in rows[0]      # row 0, col 0
    assert PLACEHOLDER + chr(d[0]) + chr(d[1]) in rows[0]      # row 0, col 1
    assert rows[1].startswith("" * 0 + PLACEHOLDER + chr(d[1]) + chr(d[0]))  # row 1, col 0
    assert out.rstrip().endswith("\x1b[39m")            # fg color restored


def test_kitty_tmux_passthrough_wrapping(monkeypatch, tmp_path):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (100, 50, 0, 0))
    plotty._cfg.update(imgcat="kitty", size=20, bg=None)
    out = plotty._render_bytes(str(_png(tmp_path)), 1)

    assert not APC.search(out)                          # no bare APCs through tmux
    n_wrapped = out.count(b"\x1bPtmux;")
    assert n_wrapped >= 2                               # delete + >=1 data chunk
    assert out.count(b"\x1b\x1b_G") == n_wrapped        # ESCs doubled inside
    # placeholders are plain text outside any passthrough envelope
    tail = out.rsplit(b"\x1b\\", 1)[1]
    assert PLACEHOLDER.encode("utf-8") in tail


def test_kitty_bg_compositing(monkeypatch, tmp_path):
    import matplotlib.image as mpimg
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (100, 50, 0, 0))
    plotty._cfg.update(imgcat="kitty", size=20, bg="#ff0000")
    out = plotty._render_bytes(str(_png(tmp_path, alpha=True)), 1)
    png = base64.standard_b64decode(b"".join(d for _, d in APC.findall(out)[1:]))
    arr = mpimg.imread(io.BytesIO(png), format="png")
    if arr.dtype != np.uint8:
        arr = (arr * 255).round().astype(np.uint8)
    assert tuple(arr[-1, 0][:3]) == (255, 0, 0)         # transparent -> bg
    assert tuple(arr[0, 0][:3]) == (10, 20, 30)         # opaque untouched


def test_resolve_imgcat_kitty_no_warning(capsys):
    assert plotty._resolve_imgcat("kitty", verbose=1) == "kitty"
    assert capsys.readouterr().err == ""                # not a "non-sixel" warning


def test_pane_render_cmd_kitty():
    plotty._cfg.update(imgcat="kitty", size=60)
    cmd = plotty._pane_render_cmd()
    assert "--render" in cmd
    assert f"{plotty._ENV}_IMGCAT=kitty" in cmd


def test_enable_kitty_outside_tmux_skips_sixel_check(monkeypatch, fake_run):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(plotty, "_stdout_supports_sixel",
                        lambda: (_ for _ in ()).throw(AssertionError("not called")))
    plotty.enable(imgcat="kitty", viewer=False, verbose=0)
    assert plotty._cfg["imgcat"] == "kitty"
    assert plotty._cfg["can_display"] is True           # explicit opt-in: trusted


def test_health_check_kitty_passthrough_warning(monkeypatch, capsys, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"-V": "tmux 3.5a", "allow-passthrough": "off"}
    plotty._cfg.update(inline=False, imgcat="kitty")
    plotty._health_check(verbose=1)
    assert "allow-passthrough" in capsys.readouterr().err


def test_health_check_kitty_quiet_when_passthrough_on(monkeypatch, capsys, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    # passthrough on; no sixel terminal feature -> still quiet (irrelevant to kitty)
    fake_run.responses = {"-V": "tmux 3.5a", "allow-passthrough": "on",
                          "display-message": "clipboard,title"}
    plotty._cfg.update(inline=False, imgcat="kitty")
    plotty._health_check(verbose=1)
    assert capsys.readouterr().err == ""
