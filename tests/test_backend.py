"""Tests for command construction, signalling, inline detection, health check."""

import os
import re
import signal
import pathlib

import plotty


# ---- per-tmux-window cache keying -------------------------------------------

def test_window_cache_keys_by_window(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"display-message": "@7\n"}    # window_id (@-prefixed)
    assert plotty._window_cache("/base") == "/base/win-7"


def test_window_cache_base_when_not_tmux(monkeypatch, fake_run):
    monkeypatch.delenv("TMUX", raising=False)
    assert plotty._window_cache("/base") == "/base"
    assert fake_run.calls == []                         # no tmux query off-tmux


def test_window_cache_scoped_to_own_window(fake_run, monkeypatch):
    # must resolve OUR window via $TMUX_PANE, not the client's focused window,
    # else a backgrounded REPL keys onto the wrong window's cache
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setenv("TMUX_PANE", "%3")
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"display-message": "@2\n"}
    assert plotty._window_cache("/base") == "/base/win-2"
    call = next(c for c in fake_run.calls if "display-message" in c)
    assert "-t" in call and "%3" in call


def test_window_cache_base_when_no_window_id(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {}                             # display-message -> ""
    assert plotty._window_cache("/base") == "/base"


def test_set_cache_repoints_path_globals(tmp_path):
    saved = (plotty._cache, plotty._last, plotty._pidfile,
             plotty._config, plotty._histdir)
    try:
        plotty._set_cache(str(tmp_path / "win-9"))
        assert plotty._cache == str(tmp_path / "win-9")
        assert plotty._pidfile == str(tmp_path / "win-9" / "viewer.pid")
        assert plotty._last == str(tmp_path / "win-9" / "last.png")
        assert os.path.isdir(plotty._cache)            # created it
    finally:
        (plotty._cache, plotty._last, plotty._pidfile,
         plotty._config, plotty._histdir) = saved


# ---- live settings (config.json) ---------------------------------------------

def test_write_config_load_settings_roundtrip(monkeypatch):
    for var in ("PLOTTY_IMGCAT", "PLOTTY_SIZE", "PLOTTY_CLEAR", "PLOTTY_BG"):
        monkeypatch.delenv(var, raising=False)
    plotty._cfg.update(imgcat="chafa --size {size}", size=42, clear=False,
                       bg="#101010")
    plotty._write_config()
    plotty._cfg.update(imgcat=None, size=60, clear=True, bg=None)  # drift
    plotty._load_settings()                       # viewer-side reload
    assert plotty._cfg["imgcat"] == "chafa --size {size}"
    assert plotty._cfg["size"] == 42
    assert plotty._cfg["clear"] is False
    assert plotty._cfg["bg"] == "#101010"


def test_load_settings_env_only_when_no_config(monkeypatch):
    monkeypatch.setenv("PLOTTY_IMGCAT", "")
    monkeypatch.setenv("PLOTTY_SIZE", "33")
    plotty._load_settings()                       # no config.json present
    assert plotty._cfg["imgcat"] is None          # "" -> built-in
    assert plotty._cfg["size"] == "33"


# ---- figure history ------------------------------------------------------------

def _publish_n(n, monkeypatch):
    monkeypatch.setattr(plotty, "_signal_viewer", lambda: True)

    class FakeFig:
        def savefig(self, path, **kw):
            with open(path, "wb") as f:
                f.write(os.urandom(8))

    plotty._cfg.update(dpi=None, inline=False)
    for _ in range(n):
        plotty._display_figure(FakeFig())


def test_history_records_and_prunes(monkeypatch):
    plotty._cfg["hist"] = 2
    _publish_n(3, monkeypatch)
    files = plotty._hist_files()
    assert len(files) == 2                        # pruned to the ring size
    assert files == sorted(files, reverse=True)   # newest first
    # newest snapshot has the same content as last.png
    assert open(files[0], "rb").read() == open(plotty._last, "rb").read()


def test_history_disabled_when_zero(monkeypatch):
    plotty._cfg["hist"] = 0
    _publish_n(2, monkeypatch)
    assert plotty._hist_files() == []


# ---- viewer pane tracking / restart-on-move ------------------------------------

def test_read_viewer_pane():
    with open(plotty._pidfile, "w") as f:
        f.write("12345\n%9\n")
    assert plotty._read_pid() == 12345
    assert plotty._read_viewer_pane() == "%9"


def test_ensure_viewer_restarts_when_pane_moved(fake_run, monkeypatch):
    fake_run.responses = {"ps -p": "python /x/plotty.py --view\n"}
    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))
        if sig == 0:
            return
    monkeypatch.setattr(plotty.os, "kill", fake_kill)
    with open(plotty._pidfile, "w") as f:
        f.write(f"{os.getpid()}\n%3\n")           # viewer lives in %3
    plotty._cfg.update(imgcat=None, pane="%9", size=60, tmux="tmux")
    plotty._ensure_viewer()                       # target is %9 now
    assert (os.getpid(), signal.SIGTERM) in killed
    assert any("send-keys" in " ".join(c) for c in fake_run.calls)


def test_ensure_viewer_keeps_viewer_in_same_pane(fake_run, monkeypatch):
    fake_run.responses = {"ps -p": "python /x/plotty.py --view\n"}
    monkeypatch.setattr(plotty.os, "kill", lambda pid, sig: None)
    with open(plotty._pidfile, "w") as f:
        f.write(f"{os.getpid()}\n%9\n")
    plotty._cfg.update(imgcat=None, pane="%9", size=60, tmux="tmux")
    plotty._ensure_viewer()
    assert not any("send-keys" in " ".join(c) for c in fake_run.calls)


# ---- status / disable / show / save --------------------------------------------

def test_status_prints_summary(monkeypatch, capsys, fake_run):
    monkeypatch.delenv("TMUX", raising=False)
    plotty._cfg.update(inline=True, can_display=True, imgcat=None, size=60,
                       dpi=None, bg=None)
    plotty.status()
    out = capsys.readouterr().out
    assert "mode:" in out and "inline -> stdout" in out
    assert "built-in sixel encoder" in out
    assert "viewer:    not running" in out
    assert plotty.__version__ in out


def test_disable_closes_auto_created_pane(fake_run, capsys):
    plotty._cfg.update(made_pane="%7", tmux="tmux")
    plotty.disable(close_pane=True, verbose=1)
    assert any(c[:3] == ["tmux", "kill-pane", "-t"] and c[3] == "%7"
               for c in fake_run.calls)
    assert plotty._cfg["made_pane"] is None
    assert "disabled" in capsys.readouterr().err


def test_disable_leaves_user_panes_alone(fake_run):
    plotty._cfg.update(made_pane=None, tmux="tmux")
    plotty.disable(close_pane=True, verbose=0)
    assert not any("kill-pane" in " ".join(c) for c in fake_run.calls)


def test_disable_closes_auto_created_pane_by_default(fake_run):
    plotty._cfg.update(made_pane="%7", tmux="tmux")
    plotty.disable(verbose=0)                       # no close_pane arg -> defaults True
    assert any(c[:3] == ["tmux", "kill-pane", "-t"] and c[3] == "%7"
               for c in fake_run.calls)
    assert plotty._cfg["made_pane"] is None


def test_disable_close_pane_false_keeps_pane(fake_run):
    plotty._cfg.update(made_pane="%7", tmux="tmux")
    plotty.disable(close_pane=False, verbose=0)     # opt out: keep the pane
    assert not any("kill-pane" in " ".join(c) for c in fake_run.calls)
    assert plotty._cfg["made_pane"] == "%7"


def test_show_explicit_figure(monkeypatch):
    shown = []
    monkeypatch.setattr(plotty, "_display_figure", lambda f: shown.append(f))
    sentinel = object()
    plotty.show(sentinel)                          # non-pyplot figure
    assert shown == [sentinel]


def test_save_copies_last_png(tmp_path):
    with open(plotty._last, "wb") as f:
        f.write(b"PNGDATA")
    dst = plotty.save(str(tmp_path / "out.png"))
    assert open(dst, "rb").read() == b"PNGDATA"


def test_save_raises_without_figure(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        plotty.save(str(tmp_path / "out.png"))


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


def test_split_direction_wide_pane_beside(fake_run):
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"display-message": "80 24\n"}   # wide
    assert plotty._split_direction("%5") == "-h"


def test_split_direction_tall_pane_below(fake_run):
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"display-message": "40 60\n"}   # tall (60*2 > 40)
    assert plotty._split_direction("%5") == "-v"


def test_split_direction_defaults_h_when_unknown(fake_run):
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {}                                # no size reported
    assert plotty._split_direction("%5") == "-h"


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


# ---- dedicated plot pane (target_pane="auto") -------------------------------

def test_find_dedicated_pane_returns_tagged(fake_run):
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"list-panes": "%10 \n%11 1\n%12 \n"}   # %11 is tagged
    assert plotty._find_dedicated_pane() == "%11"


def test_find_dedicated_pane_scoped_to_own_window(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX_PANE", "%3")
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"list-panes": "%3 \n%4 1\n"}
    assert plotty._find_dedicated_pane() == "%4"
    call = next(c for c in fake_run.calls if "list-panes" in c)
    assert "-t" in call and "%3" in call           # scoped to the REPL's window


def test_find_dedicated_pane_none_when_untagged(fake_run):
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"list-panes": "%10 \n%11 \n"}          # nothing tagged
    assert plotty._find_dedicated_pane() is None


def test_ensure_dedicated_pane_reuses_existing(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX_PANE", "%5")
    fake_run.responses = {"list-panes": "%5 \n%9 1\n"}           # %9 already plotty's
    plotty._cfg.update(tmux="tmux", made_pane=None)
    assert plotty._ensure_dedicated_pane(verbose=0) == "%9"
    assert plotty._cfg["made_pane"] == "%9"      # owned: disable(close_pane=True) closes it
    assert not any("split-window" in " ".join(c) for c in fake_run.calls)


def test_ensure_dedicated_pane_splits_and_tags_when_missing(fake_run, monkeypatch, capsys):
    monkeypatch.setenv("TMUX_PANE", "%5")
    fake_run.responses = {"list-panes": "%5 \n", "split-window": "%7\n"}
    plotty._cfg.update(tmux="tmux", made_pane=None)
    pane = plotty._ensure_dedicated_pane(verbose=1)
    assert pane == "%7"
    assert plotty._cfg["made_pane"] == "%7"
    assert plotty._cfg["can_display"] is True
    split = next(c for c in fake_run.calls if "split-window" in c)
    assert "-d" in split                         # don't steal focus from the REPL
    tag = next(c for c in fake_run.calls if "set" in c and plotty._DEDICATED_OPT in c)
    assert tag[-2:] == [plotty._DEDICATED_OPT, "1"] and "%7" in tag
    assert "created dedicated plot pane" in capsys.readouterr().err


def test_ensure_dedicated_pane_failure_disables_display(fake_run, monkeypatch, capsys):
    monkeypatch.setenv("TMUX_PANE", "%5")
    fake_run.responses = {"list-panes": "%5 \n"}   # split-window returns nothing
    plotty._cfg.update(tmux="tmux", can_display=True)
    plotty._ensure_dedicated_pane(verbose=1)
    assert plotty._cfg["can_display"] is False
    assert "cannot display figures" in capsys.readouterr().err


def test_resolve_display_pane_auto_uses_dedicated(fake_run, monkeypatch):
    monkeypatch.delenv("PLOTTY_PANE", raising=False)
    monkeypatch.setenv("TMUX_PANE", "%5")
    fake_run.responses = {"list-panes": "%5 \n%9 1\n"}
    assert plotty._resolve_display_pane("auto", verbose=0) == "%9"


def test_resolve_display_pane_explicit_overrides_auto(fake_run):
    # an explicit int takes the classic path (positive index = passthrough),
    # never the dedicated-pane logic
    assert plotty._resolve_display_pane(2, verbose=0) == "2"
    assert not any("split-window" in " ".join(c) for c in fake_run.calls)


def test_resolve_display_pane_env_overrides_auto(fake_run, monkeypatch):
    monkeypatch.setenv("PLOTTY_PANE", "3")         # PLOTTY_PANE overrides "auto"
    assert plotty._resolve_display_pane("auto", verbose=0) == "3"


# ---- recreate the dedicated pane when the user kills it ----------------------

def test_pane_alive_true_false(fake_run):
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"display-message": "%9\n"}
    assert plotty._pane_alive("%9") is True
    fake_run.responses = {}                         # default "" -> pane is gone
    assert plotty._pane_alive("%9") is False


def test_refresh_recreates_killed_dedicated_pane(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setenv("TMUX_PANE", "%5")
    # display-message (the alive-probe) returns nothing -> the old pane is dead;
    # list-panes has nothing tagged -> split a fresh one (%7)
    fake_run.responses = {"split-window": "%7\n", "list-panes": "%5 \n"}
    plotty._cfg.update(tmux="tmux", auto_pane=True, inline=False, pane="%9",
                       can_display=True, made_pane=None, verbose=0)
    assert plotty._refresh_display_pane() is True
    assert plotty._cfg["pane"] == "%7"
    assert any("split-window" in " ".join(c) for c in fake_run.calls)
    assert any("send-keys" in " ".join(c) for c in fake_run.calls)   # relaunched viewer


def test_refresh_noop_when_pane_alive(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"display-message": "%9\n"}    # still alive
    plotty._cfg.update(tmux="tmux", auto_pane=True, inline=False, pane="%9")
    assert plotty._refresh_display_pane() is False
    assert not any("split-window" in " ".join(c) for c in fake_run.calls)


def test_refresh_noop_for_explicit_target(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    plotty._cfg.update(tmux="tmux", auto_pane=False, pane="%9")
    assert plotty._refresh_display_pane() is False
    assert fake_run.calls == []                         # user's pane: never touched


# ---- user quit (Ctrl+C / q) closes plotty's own pane ------------------------

def test_own_pane_is_dedicated(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX_PANE", "%7")
    plotty._cfg.update(tmux="tmux")
    fake_run.responses = {"show": "1\n"}
    assert plotty._own_pane_is_dedicated() is True
    fake_run.responses = {"show": "\n"}                 # tag unset (user's pane)
    assert plotty._own_pane_is_dedicated() is False


def test_own_pane_is_dedicated_no_pane(monkeypatch):
    monkeypatch.delenv("TMUX_PANE", raising=False)
    assert plotty._own_pane_is_dedicated() is False


def test_kill_own_pane(fake_run, monkeypatch):
    monkeypatch.setenv("TMUX_PANE", "%7")
    plotty._cfg.update(tmux="tmux")
    plotty._kill_own_pane()
    assert any(c[:2] == ["tmux", "kill-pane"] and "%7" in c for c in fake_run.calls)


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


def test_enable_default_blocks_terminal_with_no_graphics(monkeypatch, fake_run):
    # IDE console: neither sixel nor kitty -> detection falls back to sixel and
    # the no-garbage gate still disables display
    _pin_plain_terminal_env(monkeypatch)
    monkeypatch.delenv("PLOTTY_IMGCAT", raising=False)
    monkeypatch.setattr(plotty, "_probe_terminal", lambda: (False, False))
    plotty.enable(viewer=False, verbose=0)
    assert plotty._cfg["imgcat"] is None
    assert plotty._cfg["can_display"] is False


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


def test_enable_default_is_builtin_even_with_tools_installed(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.delenv("PLOTTY_IMGCAT", raising=False)
    monkeypatch.setattr(plotty.shutil, "which", lambda c: f"/usr/bin/{c}")
    fake_run.responses = {"display-message": "sixel,clipboard"}
    plotty.enable(viewer=False, verbose=0)        # no imgcat, chafa "installed"
    assert plotty._cfg["imgcat"] is None          # built-in sixel is the default


def test_detect_renderer_tmux_sixel_terminal(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"display-message": "sixel,clipboard,title"}
    assert plotty._detect_renderer() is None       # sixel terminal -> sixel


def test_detect_renderer_tmux_no_sixel_picks_kitty(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"display-message": "clipboard,title"}
    assert plotty._detect_renderer() == "kitty"


def test_detect_renderer_tmux_ghostty_name_beats_sixel_features(monkeypatch, fake_run):
    # a `terminal-features ',*:sixel'` override (standard nested-tmux setup)
    # makes tmux claim sixel for EVERY client — but ghostty can't render it.
    # The client's terminal name must win over the poisoned features signal.
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    fake_run.responses = {"client_termname": "xterm-ghostty\n",
                          "display-message": "clipboard,sixel,title"}
    assert plotty._detect_renderer() == "kitty"


def test_detect_renderer_tmux_unknown_defaults_sixel(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")   # all queries return nothing
    assert plotty._detect_renderer() is None


def _pin_plain_terminal_env(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)


def test_detect_renderer_outside_tmux(monkeypatch):
    _pin_plain_terminal_env(monkeypatch)
    monkeypatch.setattr(plotty, "_probe_terminal", lambda: (True, True))
    assert plotty._detect_renderer() is None       # sixel wins when available
    monkeypatch.setattr(plotty, "_probe_terminal", lambda: (False, True))
    assert plotty._detect_renderer() == "kitty"    # kitty-only terminal
    monkeypatch.setattr(plotty, "_probe_terminal", lambda: (None, None))
    assert plotty._detect_renderer() is None       # unknown -> sixel default


def test_detect_renderer_outside_tmux_env_fast_path(monkeypatch):
    boom = lambda: (_ for _ in ()).throw(AssertionError("must not probe"))
    _pin_plain_terminal_env(monkeypatch)
    monkeypatch.setattr(plotty, "_probe_terminal", boom)
    monkeypatch.setenv("TERM", "xterm-ghostty")
    assert plotty._detect_renderer() == "kitty"
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    assert plotty._detect_renderer() == "kitty"
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    assert plotty._detect_renderer() == "kitty"


def test_enable_auto_detects_kitty_terminal(monkeypatch, fake_run, capsys):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.delenv("PLOTTY_IMGCAT", raising=False)
    fake_run.responses = {"display-message": "clipboard,title",
                          "allow-passthrough": "on", "-V": "tmux 3.5a"}
    plotty.enable(imgcat="auto", viewer=False, verbose=1)
    assert plotty._cfg["imgcat"] == "kitty"
    assert "kitty-graphics encoder" in capsys.readouterr().err


def test_enable_imgcat_shorthand_resolves_template(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setattr(plotty.shutil, "which", lambda c: f"/usr/bin/{c}")
    plotty.enable(imgcat="img2sixel", viewer=False, verbose=0)
    assert plotty._cfg["imgcat"] == "img2sixel -w {width}"
    plotty.enable(imgcat="chafa", viewer=False, verbose=0)
    assert plotty._cfg["imgcat"] == "chafa -f sixels --size {size}"


def test_enable_imgcat_shorthand_missing_tool_falls_back(monkeypatch, fake_run, capsys):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setattr(plotty.shutil, "which", lambda c: None)
    plotty.enable(imgcat="chafa", viewer=False, verbose=1)
    assert plotty._cfg["imgcat"] is None          # built-in fallback
    assert "not found on PATH" in capsys.readouterr().err


def test_enable_imgcat_custom_command_passthrough(monkeypatch, fake_run):
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    plotty.enable(imgcat="img2sixel -w 500 -d atkinson", viewer=False, verbose=0)
    assert plotty._cfg["imgcat"] == "img2sixel -w 500 -d atkinson"


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


def test_enable_env_fallbacks_and_param_precedence(monkeypatch, fake_run):
    # None-valued enable() args fall back to PLOTTY_* env vars; explicit args win
    monkeypatch.setenv("TMUX", "/tmp/fake,1,0")
    monkeypatch.setenv("PLOTTY_DPI", "180")
    monkeypatch.setenv("PLOTTY_SIZE", "42")
    plotty.enable(imgcat="builtin", viewer=False, verbose=0)
    assert plotty._cfg["dpi"] == "180"
    assert str(plotty._cfg["size"]) == "42"
    plotty.enable(imgcat="builtin", dpi=300, size=70, viewer=False, verbose=0)
    assert plotty._cfg["dpi"] == 300
    assert plotty._cfg["size"] == 70


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
