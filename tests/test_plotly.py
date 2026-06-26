"""plotly support: PNG byte path, figure detection, and renderer wiring.

The byte/detection tests are dependency-free (they exercise plotty's own glue);
the round-trip and renderer tests need a real plotly (+ kaleido) and skip when it
isn't installed.
"""

import os
import sys
import time
import signal
import subprocess

import pytest

import plotty


# ---- the format-agnostic byte path (no plotly needed) --------------------------

def test_publish_bytes_writes_last_png_and_history(monkeypatch):
    plotty._cfg["hist"] = 5
    data = b"\x89PNG\r\n\x1a\n" + os.urandom(16)
    assert plotty._publish_bytes(data) is True
    assert open(plotty._last, "rb").read() == data
    snaps = plotty._hist_files()
    assert snaps and open(snaps[0], "rb").read() == data


def test_display_bytes_presents(monkeypatch):
    seen = []
    monkeypatch.setattr(plotty, "_present", lambda: seen.append(True))
    plotty._cfg["can_display"] = True
    data = b"\x89PNG\r\n\x1a\n" + os.urandom(8)
    plotty._display_bytes(data)
    assert seen == [True]
    assert open(plotty._last, "rb").read() == data


def test_display_bytes_skipped_without_target(monkeypatch):
    seen = []
    monkeypatch.setattr(plotty, "_present", lambda: seen.append(True))
    monkeypatch.setattr(plotty, "_publish_bytes",
                        lambda d: seen.append("published") or True)
    plotty._cfg["can_display"] = False
    plotty._display_bytes(b"...")
    assert seen == []                             # no publish, no present


# ---- plotly figure detection ---------------------------------------------------

class _StubPlotlyFig:
    def to_dict(self):
        return {"data": [], "layout": {}}


_StubPlotlyFig.__module__ = "plotly.graph_objs._figure"


class _StubMplFig:
    def savefig(self, path, **kw):
        pass


def test_is_plotly_fig_true_for_plotly_like():
    assert plotty._is_plotly_fig(_StubPlotlyFig()) is True


def test_is_plotly_fig_false_for_matplotlib():
    assert plotty._is_plotly_fig(_StubMplFig()) is False


def test_is_plotly_fig_false_for_plain_object():
    assert plotty._is_plotly_fig(object()) is False


def test_show_routes_plotly_through_bytes(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n" + os.urandom(12)
    monkeypatch.setattr(plotty, "_plotly_png", lambda fig: png)
    presented = []
    monkeypatch.setattr(plotty, "_present", lambda: presented.append(True))
    plotty._cfg["can_display"] = True
    plotty.show(_StubPlotlyFig())
    assert presented == [True]
    assert open(plotty._last, "rb").read() == png


def test_show_skips_when_export_fails(monkeypatch):
    monkeypatch.setattr(plotty, "_plotly_png", lambda fig: None)
    presented = []
    monkeypatch.setattr(plotty, "_present", lambda: presented.append(True))
    plotty.show(_StubPlotlyFig())
    assert presented == []                        # export failed -> nothing drawn


# ---- real plotly (skipped when not installed) ----------------------------------

def test_plotly_png_roundtrip():
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    import plotly.graph_objects as go

    plotty._cfg["plotly_scale"] = 1
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 1, 2]))
    data = plotty._plotly_png(fig)
    assert data is not None
    assert data[:8] == b"\x89PNG\r\n\x1a\n"       # a real PNG came back


def test_register_and_unregister_plotly():
    pytest.importorskip("plotly")
    import plotly.io as pio

    before = pio.renderers.default
    try:
        plotty._register_plotly()
        assert plotty._plotly_on is True
        assert pio.renderers.default == "plotty"
        assert "plotty" in pio.renderers
    finally:
        plotty._unregister_plotly()
    assert plotty._plotly_on is False
    assert pio.renderers.default == before


def test_persistent_kaleido_server_lifecycle():
    pytest.importorskip("kaleido")
    assert plotty._plotly_server is False
    try:
        plotty._ensure_plotly_server()
        assert plotty._plotly_server is True
        plotty._ensure_plotly_server()            # idempotent
        assert plotty._plotly_server is True
    finally:
        plotty._stop_plotly_server()
    assert plotty._plotly_server is False


def test_plotly_server_opt_out(monkeypatch):
    pytest.importorskip("kaleido")
    monkeypatch.setenv("PLOTTY_PLOTLY_SERVER", "0")
    plotty._ensure_plotly_server()
    assert plotty._plotly_server is False         # opted out: per-call rendering


# ---- no orphaned Chrome on sudden parent death ---------------------------------
#
# The persistent server keeps a headless Chrome alive across renders. If the REPL
# is hard-killed (SIGKILL / crash / dropped SSH), atexit and disable() never run,
# so the only thing standing between us and an orphaned, memory-leaking Chrome is
# kaleido/choreographer's pipe transport tearing the browser down when the parent
# dies. This test pins that guarantee: boot the server in a detached child, kill
# the child with an uncatchable SIGKILL, and assert every process it spawned is
# gone. It cleans up any survivor itself, so a regression can never leak from CI.

_CHILD_SRC = """
import sys, time
sys.path.insert(0, {src!r})
import plotty
plotty._ensure_plotly_server()
import plotly.graph_objects as go
plotty._plotly_png(go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 9])))
print("READY", plotty._plotly_server, flush=True)
time.sleep(600)
"""


def _choreo_pids(child_src_marker):
    """pids of kaleido/choreographer processes (the helper + every Chrome), found
    by binary path so detached Chrome children (reparented to init) are caught."""
    out = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "choreographer" in cmd and child_src_marker not in cmd:
            out.add(int(pid))
    return out


def _alive(pid):
    return os.path.isdir(f"/proc/{pid}")


def test_no_orphaned_chrome_after_parent_sigkill(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    if not sys.platform.startswith("linux") or not os.path.isdir("/proc"):
        pytest.skip("process-tree assertions need Linux /proc")

    src = os.path.dirname(os.path.dirname(plotty.__file__))
    child_py = tmp_path / "kaleido_child.py"
    child_py.write_text(_CHILD_SRC.format(src=src))
    marker = str(child_py)

    before = _choreo_pids(marker)
    # start_new_session=True detaches the child like a separate REPL; we then
    # SIGKILL only its pid (the worst case — a group kill would obviously clean up).
    proc = subprocess.Popen([sys.executable, str(child_py)], start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready = False
        deadline = time.time() + 60
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line.startswith("READY"):
                ready = True
                break
        if not ready:
            pytest.skip(f"kaleido/Chrome did not boot here: {proc.stderr.read()[:300]}")

        time.sleep(1.0)                              # let Chrome finish forking helpers
        spawned = _choreo_pids(marker) - before
        if not spawned:
            pytest.skip("no Chrome processes were spawned (headless boot unavailable)")
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)        # sudden death: no cleanup hooks run
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    # A pipe-transport Chrome should notice the broken parent pipe and exit fast.
    end = time.time() + 20
    while time.time() < end and any(_alive(p) for p in spawned):
        time.sleep(0.5)

    survivors = [p for p in spawned if _alive(p)]
    for p in survivors:                              # never let the test itself leak
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass

    assert not survivors, (
        f"{len(survivors)} kaleido/Chrome process(es) orphaned after parent "
        f"SIGKILL: {survivors} — the persistent server leaks on sudden death"
    )
