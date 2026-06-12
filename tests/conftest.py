"""Shared test fixtures.

The cache dir is redirected to a temp location *before* importing plotty, so the
module-level paths (`_cache`, `_last`, `_pidfile`) never touch the real
`~/.cache/plotty`. tmux/renderer calls are stubbed via `fake_run`.
"""

import os
import sys
import types
import shutil
import tempfile
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["PLOTTY_CACHE"] = tempfile.mkdtemp(prefix="plotty-test-cache-")

import pytest  # noqa: E402
import plotty  # noqa: E402


class FakeRun:
    """Stand-in for subprocess.run: records argv, returns canned stdout.

    `responses` maps an argv substring -> stdout; first match wins, else
    `default`. Mirrors how plotty shells out to tmux / renderers.
    """

    def __init__(self):
        self.calls = []
        self.responses = {}
        self.default = ""

    def __call__(self, args, **kwargs):
        self.calls.append([str(a) for a in args])
        joined = " ".join(str(a) for a in args)
        out = self.default
        for key, val in self.responses.items():
            if key in joined:
                out = val
                break
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)


@pytest.fixture
def fake_run(monkeypatch):
    fr = FakeRun()
    monkeypatch.setattr(plotty.subprocess, "run", fr)
    return fr


@pytest.fixture(autouse=True)
def _isolate():
    """Snapshot/restore _cfg and clear stray cache files around each test."""
    snap = dict(plotty._cfg)
    plotty._warned.clear()
    plotty._term_probe = None       # re-probe per test (it's monkeypatched anyway)
    for path in (plotty._pidfile, plotty._last, plotty._config):
        try:
            os.remove(path)
        except OSError:
            pass
    shutil.rmtree(plotty._histdir, ignore_errors=True)
    yield
    plotty._cfg.clear()
    plotty._cfg.update(snap)
