"""Tests for command construction, signalling, inline detection, health check."""

import os
import re
import signal
import pathlib

import plotty


# ---- __version__ --------------------------------------------------------------

def test_version_matches_pyproject():
    root = pathlib.Path(plotty.__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "pyproject.toml has no version"
    assert plotty.__version__ == m.group(1)


# ---- renderer detection (sixel-only) ----------------------------------------

def test_is_sixel():
    assert plotty._is_sixel("chafa -f sixels --size 60")
    assert plotty._is_sixel("img2sixel -w 600")
    assert plotty._is_sixel("magick {} -resize 600x sixel:-")
    assert not plotty._is_sixel("kitten icat")
    assert not plotty._is_sixel("")
    assert not plotty._is_sixel(None)


def test_candidates_are_all_sixel():
    assert plotty._CANDIDATES                      # non-empty
    assert all(plotty._is_sixel(c) for c in plotty._CANDIDATES)


def test_enable_warns_on_non_sixel_imgcat(monkeypatch, capsys):
    monkeypatch.delenv("TMUX", raising=False)
    plotty.enable(imgcat="kitten icat", inline=True, viewer=False, verbose=1)
    assert "not sixel" in capsys.readouterr().err


# ---- _pane_render_cmd -------------------------------------------------------

def test_pane_render_cmd_external():
    plotty._cfg["imgcat"] = "chafa -f sixels --size 60"
    cmd = plotty._pane_render_cmd()
    assert cmd.startswith("chafa -f sixels --size 60 ")
    assert cmd.rstrip().endswith(plotty._last)


def test_pane_render_cmd_builtin():
    plotty._cfg["imgcat"] = None
    plotty._cfg["size"] = 60
    cmd = plotty._pane_render_cmd()
    assert "--render" in cmd
    assert f"{plotty._ENV}_CACHE=" in cmd
    assert f"{plotty._ENV}_SIZE=" in cmd
    assert "chafa" not in cmd


def test_pane_render_cmd_external_applies_width(fake_run, tmp_path, monkeypatch):
    tty = tmp_path / "tty"
    tty.write_bytes(b"")
    fake_run.responses = {"display-message": str(tty)}
    monkeypatch.setattr(plotty, "_winsize", lambda fd: (80, 24, 0, 0))
    plotty._cfg.update(imgcat="img2sixel -w {width}", pane="%2", size=40)
    cmd = plotty._pane_render_cmd()
    assert "-w 400" in cmd and "{width}" not in cmd          # 40 cells * 10px


# ---- _emit (send-keys fallback) --------------------------------------------

def test_emit_sends_clear_and_command(fake_run):
    plotty._cfg.update(imgcat="img2sixel", pane="%7", clear=True, tmux="tmux")
    plotty._emit()
    assert len(fake_run.calls) == 1
    call = fake_run.calls[0]
    assert call[:4] == ["tmux", "send-keys", "-t", "%7"]
    assert call[4].startswith("clear && img2sixel ")
    assert call[5] == "Enter"


def test_emit_without_clear(fake_run):
    plotty._cfg.update(imgcat="img2sixel", pane="%7", clear=False, tmux="tmux")
    plotty._emit()
    assert not fake_run.calls[0][4].startswith("clear && ")


# ---- _ensure_viewer ---------------------------------------------------------

def test_ensure_viewer_external_passes_imgcat_and_size(fake_run, monkeypatch):
    monkeypatch.setattr(plotty, "_read_pid", lambda: None)
    plotty._cfg.update(imgcat="chafa -f sixels --size 60", pane="%1", size=60, tmux="tmux")
    plotty._ensure_viewer()
    launch = fake_run.calls[0][4]
    assert f"{plotty._ENV}_IMGCAT=" in launch
    assert f"{plotty._ENV}_SIZE=" in launch
    assert launch.endswith("--view")


def test_ensure_viewer_builtin_passes_empty_imgcat(fake_run, monkeypatch):
    monkeypatch.setattr(plotty, "_read_pid", lambda: None)
    plotty._cfg.update(imgcat=None, pane="%1", size=60, tmux="tmux")
    plotty._ensure_viewer()
    launch = fake_run.calls[0][4]
    # built-in => pass an explicit empty IMGCAT so the viewer also uses built-in
    assert f"{plotty._ENV}_IMGCAT=''" in launch
    assert f"{plotty._ENV}_SIZE=" in launch
    assert launch.endswith("--view")


# ---- _resolve_pane ----------------------------------------------------------

def test_resolve_pane_positive_passthrough():
    assert plotty._resolve_pane(2) == "2"


def test_resolve_pane_named_passthrough():
    assert plotty._resolve_pane("Plots:0.1") == "Plots:0.1"


def test_resolve_pane_negative_indexes_pane_ids(fake_run, monkeypatch):
    monkeypatch.delenv("TMUX_PANE", raising=False)
    fake_run.responses = {"list-panes": "%10\n%11\n%12\n"}
    assert plotty._resolve_pane(-1) == "%12"
    assert plotty._resolve_pane(-2) == "%11"


def test_resolve_pane_skips_repl_own_pane(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX_PANE", "%12")     # REPL lives in the last pane
    fake_run.responses = {"list-panes": "%10\n%11\n%12\n"}
    assert plotty._resolve_pane(-1) == "%11"   # never draw into the REPL itself


def test_ensure_separate_pane_splits_when_only_own_pane(fake_run, monkeypatch, capsys):
    monkeypatch.setenv("TMUX_PANE", "%5")
    fake_run.responses = {"list-panes": "%5\n", "split-window": "%7\n"}
    plotty._cfg.update(tmux="tmux")
    pane = plotty._ensure_separate_pane(-1, verbose=1)
    assert pane == "%7"
    assert plotty._cfg["can_display"] is True
    split = next(c for c in fake_run.calls if "split-window" in c)
    assert "-d" in split                       # don't steal focus from the REPL
    assert "created plot pane" in capsys.readouterr().err


def test_ensure_separate_pane_failure_disables_display(fake_run, monkeypatch, capsys):
    monkeypatch.setenv("TMUX_PANE", "%5")
    fake_run.responses = {"list-panes": "%5\n"}   # split-window returns nothing
    plotty._cfg.update(tmux="tmux", can_display=True)
    pane = plotty._ensure_separate_pane(-1, verbose=1)
    assert pane == "%5"
    assert plotty._cfg["can_display"] is False
    assert "cannot display figures" in capsys.readouterr().err


def test_display_figure_skipped_when_cannot_display(monkeypatch, capsys):
    plotty._cfg.update(can_display=False, inline=False)
    published = []
    monkeypatch.setattr(plotty, "_publish", lambda fig: published.append(fig))
    plotty._display_figure(object())           # no savefig, no tmux, just a warning
    plotty._display_figure(object())
    assert published == []                     # display fully skipped
    assert capsys.readouterr().err.count("not displayed") == 1   # warn once


# ---- signalling -------------------------------------------------------------

def test_signal_viewer_no_pidfile_returns_false():
    assert plotty._signal_viewer() is False


def test_signal_viewer_dead_pid_returns_false():
    with open(plotty._pidfile, "w") as f:
        f.write("999999")             # almost certainly not a live process
    assert plotty._signal_viewer() is False


def test_signal_viewer_live_pid_sends_sigusr1(monkeypatch, fake_run):
    fake_run.responses = {"ps -p": "python /x/plotty.py --view\n"}
    sent = []

    def fake_kill(pid, sig):
        sent.append((pid, sig))
        if sig == 0:
            return                     # liveness probe: pretend alive
    monkeypatch.setattr(plotty.os, "kill", fake_kill)
    with open(plotty._pidfile, "w") as f:
        f.write(str(os.getpid()))
    assert plotty._signal_viewer() is True
    assert (os.getpid(), signal.SIGUSR1) in sent


def test_is_viewer_matches_only_viewer_commands(fake_run):
    fake_run.responses = {"ps -p": "python /x/plotty.py --view\n"}
    assert plotty._is_viewer(123) is True
    fake_run.responses = {"ps -p": "/Applications/Some.app/Contents/MacOS/Some\n"}
    assert plotty._is_viewer(123) is False


def test_signal_viewer_never_signals_recycled_pid(monkeypatch, fake_run):
    # a stale pidfile pointing at a recycled (non-viewer) pid must NOT be
    # signalled — SIGUSR1's default action would terminate an innocent process
    fake_run.responses = {"ps -p": "/Applications/Some.app/Contents/MacOS/Some\n"}
    sent = []

    def fake_kill(pid, sig):
        sent.append((pid, sig))
        if sig == 0:
            return
    monkeypatch.setattr(plotty.os, "kill", fake_kill)
    with open(plotty._pidfile, "w") as f:
        f.write(str(os.getpid()))
    assert plotty._signal_viewer() is False
    assert (os.getpid(), signal.SIGUSR1) not in sent


def test_ensure_viewer_respawns_over_recycled_pid(fake_run):
    with open(plotty._pidfile, "w") as f:
        f.write(str(os.getpid()))          # alive, but it's pytest, not a viewer
    plotty._cfg.update(imgcat=None, pane="%1", size=60, tmux="tmux")
    plotty._ensure_viewer()                # ps (via fake_run) says "not a viewer"
    assert any("send-keys" in " ".join(c) for c in fake_run.calls)


# ---- inline rendering: pane tty vs stdout -----------------------------------

def _tiny_png(path):
    import matplotlib.image as mpimg
    import numpy as np
    arr = np.zeros((12, 16, 3), np.uint8)
    arr[:, :8] = (200, 30, 30)
    mpimg.imsave(str(path), arr)


def test_pane_tty_query(fake_run):
    fake_run.responses = {"display-message": "/dev/ttys007\n"}
    assert plotty._pane_tty("%4") == "/dev/ttys007"
    assert fake_run.calls[0][:3] == ["tmux", "display-message", "-p"]


def test_write_inline_pipes_to_pane_tty_in_tmux(monkeypatch, tmp_path, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    tty = tmp_path / "pane_tty"          # a regular file stands in for the pane tty
    tty.write_bytes(b"")
    fake_run.responses = {"display-message": str(tty)}
    png = tmp_path / "f.png"
    _tiny_png(png)
    plotty._cfg.update(imgcat=None, pane="%9", clear=True, size=20)
    plotty._write_inline(str(png))
    data = tty.read_bytes()
    assert data.startswith(b"\x1b[H\x1b[2J")     # cleared the pane first
    assert b"\x1bPq" in data and data.rstrip().endswith(b"\x1b\\")


def test_write_inline_to_stdout_without_tmux(monkeypatch, tmp_path):
    import io
    monkeypatch.delenv("TMUX", raising=False)
    png = tmp_path / "f.png"
    _tiny_png(png)
    buf = io.BytesIO()

    class FakeOut:
        buffer = buf

        def fileno(self):
            raise OSError                # force the fd fallback in _out_fd()

    monkeypatch.setattr(plotty.sys, "stdout", FakeOut())
    plotty._cfg.update(imgcat=None, size=20)
    plotty._write_inline(str(png))
    out = buf.getvalue()
    assert out.startswith(b"\x1bPq") and out.endswith(b"\n")


# ---- inline detection / toggle in enable() ----------------------------------

def test_enable_inline_when_not_in_tmux(monkeypatch, fake_run):
    monkeypatch.delenv("TMUX", raising=False)
    plotty.enable(imgcat="img2sixel", viewer=True, verbose=0)
    assert plotty._cfg["inline"] is True
    assert fake_run.calls == []          # inline path touches no tmux


def test_enable_tmux_path(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    plotty.enable(imgcat="img2sixel", viewer=False, verbose=0)
    assert plotty._cfg["inline"] is False
    assert any("list-panes" in " ".join(c) for c in fake_run.calls)


def test_enable_force_inline_in_tmux(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    plotty.enable(imgcat="img2sixel", inline=True, viewer=True, verbose=0)
    assert plotty._cfg["inline"] is True
    # inline-in-tmux resolves the target pane (to find its tty) but spawns no viewer
    assert any("list-panes" in " ".join(c) for c in fake_run.calls)
    assert not any("send-keys" in " ".join(c) for c in fake_run.calls)


def test_enable_outside_tmux_forces_inline(monkeypatch, fake_run, capsys):
    # pane mode is impossible outside tmux: fall back to inline + warn instead
    # of send-keys'ing into arbitrary panes
    monkeypatch.delenv("TMUX", raising=False)
    plotty.enable(imgcat="img2sixel", inline=False, viewer=False, verbose=1)
    assert plotty._cfg["inline"] is True
    assert not any("list-panes" in " ".join(c) for c in fake_run.calls)
    assert "falling back to inline" in capsys.readouterr().err


def test_enable_auto_inline_blocks_non_sixel_terminal(monkeypatch, fake_run, capsys):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(plotty, "_stdout_supports_sixel", lambda: False)
    plotty.enable(imgcat="builtin", viewer=False, verbose=1)
    assert plotty._cfg["can_display"] is False
    assert "does not appear to support sixel" in capsys.readouterr().err


def test_enable_auto_inline_allows_sixel_terminal(monkeypatch, fake_run):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(plotty, "_stdout_supports_sixel", lambda: True)
    plotty.enable(imgcat="builtin", viewer=False, verbose=0)
    assert plotty._cfg["can_display"] is True


def test_enable_explicit_inline_skips_sixel_check(monkeypatch, fake_run):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(plotty, "_stdout_supports_sixel",
                        lambda: (_ for _ in ()).throw(AssertionError("not called")))
    plotty.enable(imgcat="builtin", inline=True, viewer=False, verbose=0)
    assert plotty._cfg["can_display"] is True   # user forced it: trust them


def test_parse_da1():
    assert plotty._parse_da1(b"\x1b[?62;4;22c") is True       # sixel advertised
    assert plotty._parse_da1(b"\x1b[?64;1;2;6;9;15;21;22c") is False
    assert plotty._parse_da1(b"\x1b[?6c") is False             # IDE-style reply
    assert plotty._parse_da1(b"") is None                      # no reply at all


def test_enable_force_builtin(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    for forced in ("builtin", "", False):
        plotty.enable(imgcat=forced, viewer=False, verbose=0)
        assert plotty._cfg["imgcat"] is None     # None == built-in encoder


def test_enable_force_builtin_via_env(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setenv("PLOTTY_IMGCAT", "builtin")
    plotty.enable(viewer=False, verbose=0)        # imgcat=None -> consult env
    assert plotty._cfg["imgcat"] is None


def test_enable_explicit_external_renderer(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    plotty.enable(imgcat="img2sixel", viewer=False, verbose=0)
    assert plotty._cfg["imgcat"] == "img2sixel"


def test_display_figure_uses_dpi(monkeypatch):
    plotty._cfg.update(dpi=222, inline=False)
    monkeypatch.setattr(plotty, "_signal_viewer", lambda: True)   # skip the tmux _emit
    captured = {}

    class FakeFig:
        def savefig(self, path, **kw):
            captured.update(kw)
            open(path, "wb").close()             # savefig writes the .part file

    plotty._display_figure(FakeFig())
    assert captured.get("dpi") == 222


def test_publish_saves_once_directly_to_last(monkeypatch):
    plotty._cfg.update(dpi=None, inline=False)
    monkeypatch.setattr(plotty, "_signal_viewer", lambda: True)
    saves = []

    class FakeFig:
        def savefig(self, path, **kw):
            saves.append((path, kw.get("format")))
            open(path, "wb").close()

    plotty._display_figure(FakeFig())
    # exactly one write, to the temp part file, with an explicit png format
    assert saves == [(plotty._last + ".part", "png")]
    assert os.path.exists(plotty._last)          # atomically published


def test_publish_failure_warns_once_and_skips_display(monkeypatch, capsys):
    plotty._cfg.update(dpi=None, inline=False)
    signaled = []
    monkeypatch.setattr(plotty, "_signal_viewer",
                        lambda: signaled.append(1) or True)

    class BadFig:
        def savefig(self, path, **kw):
            raise OSError("disk full")

    plotty._display_figure(BadFig())
    plotty._display_figure(BadFig())
    assert capsys.readouterr().err.count("cannot write") == 1   # warned exactly once
    assert signaled == []                                       # nothing displayed


def test_invalid_dpi_is_ignored_with_warning(monkeypatch, capsys):
    plotty._cfg.update(dpi="garbage", inline=False)
    monkeypatch.setattr(plotty, "_signal_viewer", lambda: True)
    captured = {}

    class FakeFig:
        def savefig(self, path, **kw):
            captured.update(kw)
            open(path, "wb").close()

    plotty._display_figure(FakeFig())
    assert "dpi" not in captured                 # bad value dropped, figure still shown
    assert "invalid dpi" in capsys.readouterr().err


def test_enable_dpi_from_env(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setenv("PLOTTY_DPI", "180")
    plotty.enable(imgcat="builtin", viewer=False, verbose=0)
    assert plotty._cfg["dpi"] == "180"


def test_enable_dpi_param_overrides_env(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setenv("PLOTTY_DPI", "180")
    plotty.enable(imgcat="builtin", dpi=300, viewer=False, verbose=0)
    assert plotty._cfg["dpi"] == 300


def test_enable_size_from_env(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setenv("PLOTTY_SIZE", "42")
    plotty.enable(imgcat="builtin", viewer=False, verbose=0)
    assert str(plotty._cfg["size"]) == "42"


def test_resolve_inline_arg_and_env(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("PLOTTY_INLINE", "0")
    assert plotty._resolve_inline(None) is False        # env overrides auto
    monkeypatch.setenv("PLOTTY_INLINE", "1")
    assert plotty._resolve_inline(None) is True
    assert plotty._resolve_inline(False) is False       # explicit arg wins
    monkeypatch.delenv("PLOTTY_INLINE", raising=False)
    monkeypatch.setenv("TMUX", "x")
    assert plotty._resolve_inline(None) is False         # in tmux -> pane mode
    monkeypatch.delenv("TMUX", raising=False)
    assert plotty._resolve_inline(None) is True          # no tmux -> inline


# ---- health check -----------------------------------------------------------

def test_health_check_inline_terminal_message(monkeypatch, capsys, fake_run):
    monkeypatch.delenv("TMUX", raising=False)
    plotty._cfg["inline"] = True
    plotty._health_check(verbose=1)
    err = capsys.readouterr().err
    assert "inline mode" in err and "this terminal" in err


def test_health_check_inline_in_tmux_message(monkeypatch, capsys, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"-V": "tmux 3.5a", "display-message": "sixel,clipboard"}
    plotty._cfg["inline"] = True
    plotty._health_check(verbose=1)
    err = capsys.readouterr().err
    assert "inline mode" in err and "the target tmux pane" in err


def test_health_check_old_tmux_warns(monkeypatch, capsys, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"-V": "tmux 3.2", "display-message": "sixel,title"}
    plotty._cfg["inline"] = False
    plotty._health_check(verbose=1)
    assert "older than 3.4" in capsys.readouterr().err


def test_health_check_missing_sixel_feature_warns(monkeypatch, capsys, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"-V": "tmux 3.5a", "display-message": "clipboard,title"}
    plotty._cfg["inline"] = False
    plotty._health_check(verbose=1)
    assert "terminal feature" in capsys.readouterr().err


def test_health_check_ok_is_quiet(monkeypatch, capsys, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"-V": "tmux 3.5a", "display-message": "sixel,clipboard"}
    plotty._cfg["inline"] = False
    plotty._health_check(verbose=1)
    assert capsys.readouterr().err == ""


def test_health_check_silent_when_not_verbose(capsys, fake_run):
    plotty._cfg["inline"] = True
    plotty._health_check(verbose=0)
    assert capsys.readouterr().err == ""


def test_tmux_version_parsing(fake_run):
    fake_run.responses = {"-V": "tmux 3.5a"}
    assert plotty._tmux_version() == (3, 5)
    fake_run.responses = {"-V": "tmux next-3.6"}
    assert plotty._tmux_version() == (3, 6)
    fake_run.default = "garbage"
    fake_run.responses = {}
    assert plotty._tmux_version() is None
