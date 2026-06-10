"""End-to-end viewer test: spawn `--view`, drive it with real signals.

Uses the built-in encoder (PLOTTY_IMGCAT=""), so it needs no external renderer.
"""

import os
import sys
import time
import signal
import subprocess

import numpy as np
import pytest

import plotty

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")


def test_viewer_signal_lifecycle(tmp_path):
    import matplotlib.image as mpimg

    cache = tmp_path / "cache"
    cache.mkdir()
    arr = np.zeros((30, 40, 3), np.uint8)
    arr[:, :20] = (200, 30, 30)
    arr[:, 20:] = (30, 30, 200)
    mpimg.imsave(str(cache / "last.png"), arr)

    env = dict(os.environ)
    env.update(PLOTTY_CACHE=str(cache), PLOTTY_IMGCAT="",
               PLOTTY_CLEAR="0", PLOTTY_SIZE="20")

    proc = subprocess.Popen([sys.executable, plotty.__file__, "--view"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    pidfile = cache / "viewer.pid"
    try:
        deadline = time.time() + 15
        while not pidfile.exists() and time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("viewer exited before writing pidfile")
            time.sleep(0.05)
        assert pidfile.exists(), "viewer did not write its pidfile"

        time.sleep(0.4)                       # let signal handlers register
        os.kill(proc.pid, signal.SIGWINCH)    # ask for a redraw
        time.sleep(0.2)
        os.kill(proc.pid, signal.SIGTERM)     # graceful shutdown
        out, _ = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 0, "viewer must always exit 0 (clean)"
    assert b"\x1bPq" in out, "viewer never emitted a sixel frame"
    assert not pidfile.exists(), "viewer did not clean up its pidfile on exit"


def test_viewer_exits_cleanly_when_pane_dies(tmp_path):
    """When the viewer's stdout (the pane pty) goes away, it must exit 0 — an
    abnormal exit triggers macOS's "Python quit unexpectedly" crash dialog."""
    import matplotlib.image as mpimg

    cache = tmp_path / "cache"
    cache.mkdir()
    arr = np.zeros((24, 32, 3), np.uint8)
    arr[:, :12] = (40, 180, 40)
    mpimg.imsave(str(cache / "last.png"), arr)

    env = dict(os.environ)
    env.update(PLOTTY_CACHE=str(cache), PLOTTY_IMGCAT="",
               PLOTTY_CLEAR="0", PLOTTY_SIZE="20")
    proc = subprocess.Popen([sys.executable, plotty.__file__, "--view"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    pidfile = cache / "viewer.pid"
    try:
        deadline = time.time() + 15
        while not pidfile.exists() and time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("viewer exited before writing pidfile")
            time.sleep(0.05)
        assert pidfile.exists(), "viewer did not write its pidfile"

        proc.stdout.close()                    # the "pane" reader goes away
        time.sleep(0.3)
        try:
            os.kill(proc.pid, signal.SIGUSR1)  # redraw -> write hits EPIPE
        except ProcessLookupError:
            pass                               # already exited: fine
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert proc.returncode == 0, "viewer must exit cleanly when its tty dies"
    assert not pidfile.exists(), "viewer did not clean up its pidfile on exit"
