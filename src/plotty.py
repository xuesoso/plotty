"""
plotty - display matplotlib figures in a dedicated tmux pane (plot + tty).

The Python/Jupyter analogue of MuxDisplay.jl, built for SSH + tmux. The backend
(this module) runs in your REPL; a tiny viewer runs in the plot pane and redraws
on SIGUSR1 (new figure) and SIGWINCH (pane resize/zoom). Only the rendered sixel
bytes cross SSH, so it works the same locally and over a remote session.

    import plotty
    plotty.enable()                      # auto-detects renderer + last pane
    plotty.enable(target_pane=2)         # or pick a pane explicitly
    plotty.disable()                     # stop the viewer + auto-display

Renderer auto-detection is sixel-only (the SSH-robust path): chafa, img2sixel,
ImageMagick. If none is on PATH it falls back to a built-in, dependency-free
sixel encoder (stdlib + numpy, which ships with matplotlib). A non-sixel command
may be passed explicitly as imgcat= but warns that it may not work over SSH.

Display modes: a viewer process running in a tmux pane (default in tmux), or
"inline" mode which renders sixel itself (no viewer) and writes it to the target
pane's tty when in tmux, or to the current terminal's stdout when not. Choose
with enable(inline=...) / PLOTTY_INLINE.

To rename this package, just rename the file: the matplotlib backend string is
derived from the module name automatically.

Config via env vars (optional; enable() args override): PLOTTY_PANE,
PLOTTY_IMGCAT, PLOTTY_CLEAR, PLOTTY_TMUX, PLOTTY_DPI, PLOTTY_CLOSE, PLOTTY_CACHE,
PLOTTY_SIZE, PLOTTY_INLINE.
"""

import os
import re
import sys
import select
import signal
import shlex
import shutil
import tempfile
import itertools
import subprocess

import numpy as np
import matplotlib
from matplotlib import image as mpimg
from matplotlib._pylab_helpers import Gcf
from matplotlib.backends import backend_agg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_ENV = "PLOTTY"   # env var prefix (kept stable even if the file is renamed)

# Sixel renderer candidates, in priority order (first one found on PATH wins).
# Sixel is the only SSH-robust path, so non-sixel protocols (kitty/iTerm) are
# intentionally excluded. Placeholders are substituted at render time:
#   "{}"      -> the image path (else it's appended)
#   "{size}"  -> display width in terminal cells  (_cfg["size"])
#   "{width}" -> display width in pixels          (size cells * pane cell width)
_CANDIDATES = [
    "chafa -f sixels --size {size}",       # sizes in cells
    "img2sixel -w {width}",                # sizes in pixels
    "magick {} -resize {width}x sixel:-",  # sizes in pixels
    "convert {} -resize {width}x sixel:-",
]


def _env(key, default):
    return os.environ.get(f"{_ENV}_{key}", default)


_cfg = {
    "pane":   _env("PANE", "-1"),
    "imgcat": _env("IMGCAT", None),          # None -> auto-detect / built-in encoder
    "clear":  _env("CLEAR", "1") != "0",
    "tmux":   _env("TMUX", "tmux"),
    "dpi":    _env("DPI", None),
    "close":  _env("CLOSE", "1") != "0",
    "size":   _env("SIZE", "60"),            # max display width in terminal cells
    "inline": False,                         # set in enable(): True when not in tmux
}

_cache = os.path.expanduser(_env("CACHE", "~/.cache/plotty"))
os.makedirs(_cache, exist_ok=True)
_last = os.path.join(_cache, "last.png")
_pidfile = os.path.join(_cache, "viewer.pid")

_tmpdir = tempfile.mkdtemp(prefix="plotty-")
_counter = itertools.count()
_recent = []
_KEEP = 8


# ---- renderer detection -----------------------------------------------------

def _is_sixel(cmd):
    return bool(cmd) and "sixel" in cmd.lower()


def _auto_imgcat():
    """Return the first renderer command available on PATH, else None."""
    for cmd in _CANDIDATES:
        if shutil.which(shlex.split(cmd)[0]):
            return cmd
    return None


def _fmt(cmd, path):
    q = shlex.quote(path)
    return cmd.replace("{}", q) if "{}" in cmd else f"{cmd} {q}"


def _resolve_cmd(cmd, fd):
    """Fill renderer size placeholders: {size}=width in cells, {width}=pixels.

    {width} is derived from the target terminal (fd) so pixel-based renderers
    follow `size` too. Renderers without either placeholder are left untouched.
    """
    if not cmd:
        return cmd
    if "{size}" in cmd:
        cmd = cmd.replace("{size}", str(int(_cfg["size"])))
    if "{width}" in cmd:
        cmd = cmd.replace("{width}", str(_target_px_width(fd)))
    return cmd


# ---- built-in sixel encoder (dependency-free fallback) ----------------------
#
# Used when no external renderer (chafa/img2sixel/magick) is on PATH. The Agg
# canvas gives us RGBA pixels and numpy ships with matplotlib, so we can encode
# sixel ourselves and honour the "stdlib + matplotlib only" rule with no extra
# runtime dependency. External renderers (when present) stay the preferred path
# because they dither for higher quality.

def _out_fd():
    try:
        return sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        return 1


def _winsize(fd):
    """Return (cols, rows, xpixels, ypixels); pixels are 0 if unreported."""
    try:
        import fcntl
        import struct
        import termios
        packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8)
        rows, cols, xpix, ypix = struct.unpack("HHHH", packed)
        return cols, rows, xpix, ypix
    except Exception:
        cs = shutil.get_terminal_size((80, 24))
        return cs.columns, cs.lines, 0, 0


def _target_px_width(fd):
    """Target display width in pixels: `size` cells, capped to the pane width."""
    cols, rows, xpix, ypix = _winsize(fd)
    size = int(_cfg["size"])
    cell_w = (xpix / cols) if (xpix and cols) else 10.0
    target_cols = min(size, cols) if cols else size      # never wider than the pane
    return max(1, round(target_cols * cell_w))


def _target_size(fd, w, h):
    """Pixel size to render at: scale to `size` cells wide, fit within the pane.

    `size` cells map to pixels via the terminal's reported cell size (or a 10x20
    guess when unreported, common in tmux) and scale the image up *or* down to
    that width; the result is then bounded by the pane height.
    """
    cols, rows, xpix, ypix = _winsize(fd)
    cell_h = (ypix / rows) if (ypix and rows) else 20.0
    scale = _target_px_width(fd) / w
    max_h = max((rows or 24) - 1, 1) * cell_h
    if h * scale > max_h:                        # don't overflow the pane height
        scale = max_h / h
    return max(1, round(w * scale)), max(1, round(h * scale))


def _load_rgb(path):
    """Read a PNG into an (H, W, 3) uint8 array, compositing alpha over white."""
    a = mpimg.imread(path)                       # matplotlib reads PNG w/o Pillow
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if np.issubdtype(a.dtype, np.floating):
        a = (a * 255.0).round().clip(0, 255).astype(np.uint8)
    else:
        a = a.astype(np.uint8)
    if a.shape[2] == 4:
        alpha = a[..., 3:4].astype(np.float32) / 255.0
        rgb = a[..., :3].astype(np.float32)
        a = (rgb * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)
    return np.ascontiguousarray(a[..., :3])


def _resize(img, tw, th):
    """Nearest-neighbour resample to (th, tw)."""
    h, w = img.shape[:2]
    if tw == w and th == h:
        return img
    ys = np.clip(np.arange(th) * h // th, 0, h - 1)
    xs = np.clip(np.arange(tw) * w // tw, 0, w - 1)
    return img[ys][:, xs]


def _make_box(pixels, ids):
    px = pixels[ids]
    rng = px.max(axis=0).astype(np.int32) - px.min(axis=0).astype(np.int32)
    ch = int(rng.argmax())
    return (ids, int(rng[ch]), ch)


def _quantize(rgb, ncolors=256):
    """Median-cut quantization. Returns (palette (K,3) uint8, indices (N,) int)."""
    pixels = rgb.reshape(-1, 3)
    boxes = [_make_box(pixels, np.arange(pixels.shape[0]))]
    while len(boxes) < ncolors:
        si, best = -1, 0
        for i, (ids, rng, _) in enumerate(boxes):
            if ids.size > 1 and rng > best:
                si, best = i, rng
        if si < 0 or best == 0:
            break                                # all boxes are single-colour
        ids, _, ch = boxes.pop(si)
        ids = ids[np.argsort(pixels[ids, ch], kind="stable")]
        mid = ids.size // 2
        boxes.append(_make_box(pixels, ids[:mid]))
        boxes.append(_make_box(pixels, ids[mid:]))
    palette = np.empty((len(boxes), 3), np.uint8)
    indices = np.empty(pixels.shape[0], np.int32)
    for i, (ids, _, _) in enumerate(boxes):
        palette[i] = pixels[ids].mean(axis=0).round().astype(np.uint8)
        indices[ids] = i
    return palette, indices


def _rle(codes):
    """Run-length encode a 1-D array of sixel byte values (already offset by 63)."""
    n = codes.shape[0]
    if n == 0:
        return b""
    change = np.ones(n, dtype=bool)
    change[1:] = codes[1:] != codes[:-1]
    starts = np.flatnonzero(change)
    runs = np.diff(np.append(starts, n))
    out = bytearray()
    for val, run in zip(codes[starts].tolist(), runs.tolist()):
        if run > 3:
            out += b"!%d%c" % (run, val)
        else:
            out += bytes([val]) * run
    return bytes(out)


def _sixel_bytes(palette, indices, h, w):
    """Assemble a DCS sixel stream from a palette + per-pixel palette indices."""
    idx = indices.reshape(h, w)
    out = bytearray(b"\x1bPq")
    out += b'"1;1;%d;%d' % (w, h)                # raster attributes
    for i, c in enumerate(palette):              # palette, scaled to 0-100
        out += b"#%d;2;%d;%d;%d" % (
            i,
            round(int(c[0]) * 100 / 255),
            round(int(c[1]) * 100 / 255),
            round(int(c[2]) * 100 / 255),
        )
    first_band = True
    for top in range(0, h, 6):                   # 6 pixel rows per sixel band
        if not first_band:
            out += b"-"                          # next band
        first_band = False
        band = idx[top:top + 6]
        bh = band.shape[0]
        first_color = True
        for ci in np.unique(band):
            if not first_color:
                out += b"$"                      # overlay next colour on band
            first_color = False
            out += b"#%d" % int(ci)
            eq = band == ci
            bits = np.zeros(w, dtype=np.int64)
            for r in range(bh):
                bits |= eq[r].astype(np.int64) << r
            out += _rle(bits + 63)
    out += b"\x1b\\"
    return bytes(out)


def _render_bytes(path, fd):
    """Return the terminal byte stream to display `path` (external cmd or built-in)."""
    cmd = _cfg["imgcat"]
    if cmd:
        full = _fmt(_resolve_cmd(cmd, fd), path)
        return subprocess.run(["sh", "-c", full], capture_output=True).stdout
    img = _load_rgb(path)
    tw, th = _target_size(fd, img.shape[1], img.shape[0])
    img = _resize(img, tw, th)
    palette, indices = _quantize(img)
    return _sixel_bytes(palette, indices, img.shape[0], img.shape[1])


# ---- pane resolution --------------------------------------------------------

def _resolve_pane(target):
    """Negative ints index the current window's panes Python-style (-1 = last)."""
    try:
        idx = int(target)
    except (TypeError, ValueError):
        return str(target)                     # named target, e.g. "Plots:0.0"
    if idx >= 0:
        return str(idx)
    try:
        out = subprocess.run([_cfg["tmux"], "list-panes", "-F", "#{pane_id}"],
                             capture_output=True, text=True, check=False)
        ids = out.stdout.split()
        if ids:
            return ids[idx]                    # stable pane id (%N)
    except OSError:
        pass
    return str(target)


# ---- talking to the viewer (or send-keys fallback) --------------------------

def _read_pid():
    try:
        with open(_pidfile) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _is_viewer(pid):
    """True if pid is a live plotty viewer process.

    Pids get recycled: after an unclean shutdown a stale pidfile can point at an
    unrelated process, and signalling it (SIGUSR1's default action: terminate)
    would kill an innocent program — on macOS that pops a "quit unexpectedly"
    crash dialog. Verify the command line before ever sending a signal.
    """
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "args="],
                             capture_output=True, text=True, check=False).stdout
    except OSError:
        return False
    return "--view" in out or "plotty-view" in out


def _signal_viewer():
    pid = _read_pid()
    if _alive(pid) and _is_viewer(pid):
        try:
            os.kill(pid, signal.SIGUSR1)
            return True
        except OSError:
            return False
    return False


def _pane_render_cmd():
    """Shell command that renders last.png in the plot pane (external or built-in)."""
    cmd = _cfg["imgcat"]
    if cmd:
        fd, opened = -1, None
        if "{width}" in cmd:                     # needs the pane's pixel width
            tty = _pane_tty(_cfg["pane"])
            if tty:
                try:
                    opened = os.open(tty, os.O_RDONLY | os.O_NONBLOCK)
                    fd = opened
                except OSError:
                    pass
        resolved = _resolve_cmd(cmd, fd)
        if opened is not None:
            os.close(opened)
        return _fmt(resolved, _last)
    env = (
        f"{_ENV}_IMGCAT='' "                      # force built-in in the subprocess
        f"{_ENV}_CACHE={shlex.quote(_cache)} "
        f"{_ENV}_SIZE={shlex.quote(str(_cfg['size']))}"
    )
    return (
        f"{env} {shlex.quote(sys.executable)} "
        f"{shlex.quote(os.path.abspath(__file__))} --render"
    )


def _emit():
    """send-keys fallback: tell the pane's shell to render last.png itself."""
    cmd = _pane_render_cmd()
    if _cfg["clear"]:
        cmd = "clear && " + cmd
    subprocess.run([_cfg["tmux"], "send-keys", "-t", str(_cfg["pane"]), cmd, "Enter"],
                   check=False)


def _pane_tty(pane):
    """The tty device path of a tmux pane, e.g. /dev/ttys003 (None on failure)."""
    try:
        out = subprocess.run(
            [_cfg["tmux"], "display-message", "-p", "-t", str(pane), "#{pane_tty}"],
            capture_output=True, text=True, check=False)
    except OSError:
        return None
    tty = out.stdout.strip()
    return tty or None


def _write_inline(path):
    """Render sixel without a viewer: to the target tmux pane's tty when in tmux,
    otherwise to this terminal's own stdout."""
    try:
        if os.environ.get("TMUX") is not None:
            tty = _pane_tty(_cfg["pane"])
            if tty:
                with open(tty, "wb", buffering=0) as out:
                    data = _render_bytes(path, out.fileno())
                    if _cfg["clear"]:
                        out.write(b"\x1b[H\x1b[2J")
                    out.write(data)
                return
        data = _render_bytes(path, _out_fd())
        buf = sys.stdout.buffer
        buf.write(data)
        buf.write(b"\n")
        buf.flush()
    except Exception as exc:
        print(f"[{__name__}] inline render failed: {exc}", file=sys.stderr)


def _publish(src):
    tmp = _last + ".part"
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, _last)
    except OSError:
        pass


def _display_figure(fig):
    path = os.path.join(_tmpdir, f"fig-{next(_counter):04d}.png")
    kw = {"bbox_inches": "tight"}
    if _cfg["dpi"]:
        kw["dpi"] = int(_cfg["dpi"])
    fig.savefig(path, **kw)
    _recent.append(path)
    while len(_recent) > _KEEP:
        try:
            os.remove(_recent.pop(0))
        except OSError:
            pass
    _publish(path)
    if _cfg["inline"]:
        _write_inline(path)
    elif not _signal_viewer():
        _emit()


# ---- matplotlib backend API -------------------------------------------------

FigureCanvas = FigureCanvasAgg
FigureManager = backend_agg.FigureManagerBase


def new_figure_manager(num, *args, FigureClass=Figure, **kwargs):
    return new_figure_manager_given_figure(num, FigureClass(*args, **kwargs))


def new_figure_manager_given_figure(num, figure):
    return backend_agg.new_figure_manager_given_figure(num, figure)


def draw_if_interactive():
    pass


def show(*args, **kwargs):
    managers = Gcf.get_all_fig_managers()
    if not managers:
        return
    for manager in managers:
        _display_figure(manager.canvas.figure)
    if _cfg["close"]:
        Gcf.destroy_all()


def redraw():
    if _cfg["inline"]:
        if os.path.exists(_last):
            _write_inline(_last)
    elif not _signal_viewer() and os.path.exists(_last):
        _emit()


# ---- the viewer (runs in the plot pane) -------------------------------------

def _apply_env():
    """Load renderer settings from the environment (for the --view/--render subprocesses)."""
    _cfg["imgcat"] = _env("IMGCAT", "") or None      # empty -> built-in encoder
    _cfg["size"] = _env("SIZE", _cfg["size"])


def view():
    """Viewer loop: redraw last.png on SIGUSR1 (new figure) / SIGWINCH (resize),
    exit on SIGTERM/SIGINT/SIGHUP.

    Signal handlers do no work — the kernel writes the signal number to a
    self-pipe (signal.set_wakeup_fd) and all rendering/writing happens in the
    main loop. That avoids handler reentrancy during resize bursts and
    unguarded writes to a dying pty, and every exit path is a clean
    os._exit(0): an abnormal viewer exit (traceback or fatal signal) makes
    macOS pop a "Python quit unexpectedly" crash dialog when the surrounding
    session is torn down. Draining the pipe for ~50 ms also coalesces signal
    bursts, so a resize storm redraws once.
    """
    _apply_env()
    clear = _env("CLEAR", "1" if _cfg["clear"] else "0") != "0"

    draw_sigs = {int(getattr(signal, n)) for n in ("SIGUSR1", "SIGWINCH")
                 if hasattr(signal, n)}
    quit_sigs = {int(getattr(signal, n)) for n in ("SIGTERM", "SIGINT", "SIGHUP")
                 if hasattr(signal, n)}

    def _cleanup():
        try:
            if _read_pid() == os.getpid():
                os.remove(_pidfile)
        except OSError:
            pass

    def _draw():
        """Render last.png to stdout. False means the tty is gone: exit."""
        if not os.path.exists(_last):
            return True
        try:
            data = _render_bytes(_last, _out_fd())
        except Exception:
            return True                          # bad/partial image: skip frame
        try:
            out = sys.stdout.buffer
            if clear:
                out.write(b"\x1b[H\x1b[2J")      # home + clear screen
            out.write(data)
            out.flush()
            return True
        except OSError:                          # EPIPE/EIO: pane is gone
            return False

    try:
        if hasattr(signal, "SIGPIPE"):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        rfd, wfd = os.pipe()
        os.set_blocking(wfd, False)
        signal.set_wakeup_fd(wfd)                # kernel writes signal numbers here
        for sig in draw_sigs | quit_sigs:
            signal.signal(sig, lambda *_: None)  # neutralize default dispositions
        with open(_pidfile, "w") as f:
            f.write(str(os.getpid()))
        ok = _draw()
        while ok:
            sigs = set(os.read(rfd, 128))        # blocks: zero CPU while idle
            while select.select([rfd], [], [], 0.05)[0]:
                sigs |= set(os.read(rfd, 128))   # coalesce resize/figure bursts
            if sigs & quit_sigs:
                break
            if sigs & draw_sigs:
                ok = _draw()
    except BaseException:
        pass                                     # never crash out of the viewer
    _cleanup()
    os._exit(0)


# ---- setup / teardown -------------------------------------------------------

def _ensure_viewer():
    pid = _read_pid()
    if _alive(pid) and _is_viewer(pid):      # don't trust recycled pids
        return
    # Always pass IMGCAT (empty == built-in) so the viewer's renderer matches the
    # backend's, regardless of any PLOTTY_IMGCAT inherited by the pane's shell.
    parts = [
        f"{_ENV}_IMGCAT={shlex.quote(_cfg['imgcat'] or '')}",
        f"{_ENV}_CLEAR={'1' if _cfg['clear'] else '0'}",
        f"{_ENV}_CACHE={shlex.quote(_cache)}",
        f"{_ENV}_SIZE={shlex.quote(str(_cfg['size']))}",
    ]
    launch = (
        " ".join(parts)
        + f" {shlex.quote(sys.executable)} {shlex.quote(os.path.abspath(__file__))} --view"
    )
    subprocess.run([_cfg["tmux"], "send-keys", "-t", str(_cfg["pane"]), launch, "Enter"],
                   check=False)


_hook_cb = None


def hook():
    global _hook_cb
    if _hook_cb is not None:
        return
    try:
        ip = get_ipython()  # noqa: F821
    except NameError:
        ip = None
    if ip is not None:
        _hook_cb = lambda *a, **k: show()
        ip.events.register("post_run_cell", _hook_cb)


def _tmux_version():
    try:
        out = subprocess.run([_cfg["tmux"], "-V"], capture_output=True, text=True,
                             check=False).stdout
    except OSError:
        return None
    m = re.search(r"(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _tmux_features():
    """The terminal features tmux has resolved for the current client, if any."""
    for fmt in ("#{client_termfeatures}", "#{terminal-features}"):
        try:
            out = subprocess.run([_cfg["tmux"], "display-message", "-p", fmt],
                                 capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            return None
        if out:
            return out
    return ""


def _health_check(verbose):
    """Warn about likely sixel-display problems up front (best effort)."""
    if not verbose:
        return
    name = __name__
    intmux = os.environ.get("TMUX") is not None
    if _cfg["inline"]:
        where = "the target tmux pane" if intmux else "this terminal"
        print(f"[{name}] inline mode: piping sixel to {where} "
              f"(requires a sixel-capable terminal)", file=sys.stderr)
    elif not intmux:
        print(f"[{name}] inline mode is off but you are not in tmux; pane routing "
              f"will not work — pass inline=True to enable()", file=sys.stderr)
    if intmux:                                   # both modes lean on tmux's sixel
        ver = _tmux_version()
        if ver is not None and ver < (3, 4):
            print(f"[{name}] tmux {ver[0]}.{ver[1]} is older than 3.4 and may not "
                  f"render sixel; upgrade tmux for native sixel support",
                  file=sys.stderr)
        feats = _tmux_features()
        if feats is not None and "sixel" not in feats:
            print(f"[{name}] tmux does not report a 'sixel' terminal feature; if "
                  f"plots don't appear, run: tmux set -as terminal-features "
                  f"',*:sixel' (and make sure your terminal supports sixel)",
                  file=sys.stderr)


def _resolve_inline(inline):
    """inline: None -> auto (True when not in tmux); else honour the bool.

    `PLOTTY_INLINE` (1/0) overrides auto-detection but not an explicit argument.
    """
    if inline is not None:
        return bool(inline)
    env_inline = _env("INLINE", None)
    if env_inline is not None:
        return env_inline != "0"
    return os.environ.get("TMUX") is None


def _resolve_imgcat(imgcat, verbose):
    """Resolve the renderer command, where None means the built-in encoder.

    imgcat=None consults PLOTTY_IMGCAT then auto-detects; "" / "builtin" / False
    force the built-in encoder; any other string is used as the command.
    """
    if imgcat is None:
        imgcat = _env("IMGCAT", None)
    if imgcat in ("", "builtin", False):
        return None
    if imgcat is None:                           # auto-detect an external renderer
        imgcat = _auto_imgcat()
        if imgcat is None and verbose:
            print(f"[{__name__}] no external renderer on PATH; using built-in "
                  f"sixel encoder (install chafa for higher-quality output)",
                  file=sys.stderr)
    if imgcat and verbose and not _is_sixel(imgcat):
        print(f"[{__name__}] {shlex.split(imgcat)[0]} is not sixel, so image "
              f"display may not work over ssh", file=sys.stderr)
    return imgcat


def enable(target_pane=-1, imgcat=None, clear=True, tmux="tmux", dpi=None,
           close=True, size=None, inline=None, viewer=True, verbose=1):
    """Activate plotty: detect a renderer, point at a pane, start the viewer.

    inline=None (default) auto-selects: inline mode when not in tmux, viewer-pane
    mode when in tmux. In inline mode the backend renders sixel itself (no viewer
    process) and writes it to the target pane's tty when in tmux, or to this
    terminal's stdout when not. inline=True forces inline even inside tmux;
    inline=False forces viewer-pane mode. `PLOTTY_INLINE=1/0` sets the default.

    imgcat=None (default) auto-detects an external renderer (chafa/img2sixel/
    magick), falling back to the built-in encoder if none is found. Pass
    imgcat="builtin" (or "" / False) to force the built-in encoder even when an
    external one is installed; pass a command string to use it explicitly.
    `PLOTTY_IMGCAT` sets the default (`PLOTTY_IMGCAT=builtin` forces built-in).
    This applies to both viewer and inline modes.

    size (display width in cells, default 60) and dpi (matplotlib savefig DPI;
    None = matplotlib's own default) control display size and source-image
    resolution respectively. Both fall back to `PLOTTY_SIZE` / `PLOTTY_DPI` when
    the argument is None. Raise dpi when displaying at a large size so the source
    PNG has enough pixels to stay sharp (else the renderer upscales it).
    """
    _cfg["tmux"] = tmux
    _cfg["clear"] = clear
    _cfg["dpi"] = _env("DPI", None) if dpi is None else dpi
    _cfg["close"] = close
    _cfg["size"] = _env("SIZE", 60) if size is None else size

    _cfg["imgcat"] = _resolve_imgcat(imgcat, verbose)

    matplotlib.use(f"module://{__name__}")
    matplotlib.interactive(True)

    _cfg["inline"] = _resolve_inline(inline)
    _health_check(verbose)
    if _cfg["inline"]:
        if os.environ.get("TMUX") is not None:
            _cfg["pane"] = _resolve_pane(target_pane)   # pipe sixel to this pane's tty
    else:
        _cfg["pane"] = _resolve_pane(target_pane)
        if viewer:
            _ensure_viewer()
    hook()


def disable():
    """Stop the viewer, unhook auto-display, and quiet matplotlib output."""
    pid = _read_pid()
    if _alive(pid) and _is_viewer(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    global _hook_cb
    if _hook_cb is not None:
        try:
            get_ipython().events.unregister("post_run_cell", _hook_cb)  # noqa: F821
        except Exception:
            pass
        _hook_cb = None
    try:
        matplotlib.use("agg")
    except Exception:
        pass


if __name__ == "__main__":
    if "--view" in sys.argv:
        view()
    elif "--render" in sys.argv:
        # render last.png to this pane's stdout (send-keys fallback)
        _apply_env()
        if os.path.exists(_last):
            sys.stdout.buffer.write(_render_bytes(_last, _out_fd()))
            sys.stdout.buffer.flush()
