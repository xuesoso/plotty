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

Rendering uses the built-in, dependency-free sixel encoder by default (stdlib +
numpy, which ships with matplotlib) — no external tools needed. Opt into an
external sixel encoder with enable(imgcat="chafa") / "img2sixel" / "magick"
(slightly faster, better resampling), imgcat="auto" to pick the first one on
PATH, or pass a full custom command. A non-sixel command warns that it may not
work over SSH.

Display modes: a viewer process running in a tmux pane (default in tmux), or
"inline" mode which renders sixel itself (no viewer) and writes it to the target
pane's tty when in tmux, or to the current terminal's stdout when not. Choose
with enable(inline=...) / PLOTTY_INLINE.

plotty never draws into the pane you are typing in: if the tmux window has no
separate pane, enable() splits one off automatically; without a usable sixel
display (e.g. an IDE console), it warns instead of printing escape garbage.

To rename this package, just rename the file: the matplotlib backend string is
derived from the module name automatically.

Config via env vars (optional; enable() args override): PLOTTY_PANE,
PLOTTY_IMGCAT, PLOTTY_CLEAR, PLOTTY_TMUX, PLOTTY_DPI, PLOTTY_CLOSE, PLOTTY_CACHE,
PLOTTY_SIZE, PLOTTY_INLINE, PLOTTY_BG, PLOTTY_HIST.

In the viewer pane: p/k = previous figure, n/j = next, q = quit. Re-running
enable() with new settings updates a running viewer live.
"""

import os
import re
import sys
import json
import time
import select
import signal
import shlex
import shutil
import subprocess

import numpy as np
import matplotlib
from matplotlib import image as mpimg
from matplotlib._pylab_helpers import Gcf
from matplotlib.backends import backend_agg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def _find_version():
    """Resolve __version__ with pyproject.toml as the single source of truth:
    read it directly in a source checkout, else ask the installed package
    metadata (which setuptools filled from pyproject.toml at build time)."""
    try:
        pyproject = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, "pyproject.toml")
        with open(pyproject) as f:
            m = re.search(r'^version\s*=\s*"([^"]*)"', f.read(), re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version          # Python 3.8+
        return version(__name__)
    except Exception:
        pass
    try:
        import pkg_resources                            # Python 3.7 fallback
        return pkg_resources.get_distribution(__name__).version
    except Exception:
        pass
    return "0+unknown"


__version__ = _find_version()

_ENV = "PLOTTY"   # env var prefix (kept stable even if the file is renamed)

# External sixel renderer candidates (opt-in: imgcat="auto" picks the first on
# PATH; a bare tool name like imgcat="chafa" selects its template). Sixel is the
# only SSH-robust path, so non-sixel protocols (kitty/iTerm) are intentionally
# excluded. Placeholders are substituted at render time:
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
    "bg":     _env("BG", None),              # '#rrggbb' alpha-composite background
    "hist":   _env("HIST", "10"),            # figures kept for viewer history keys
    "inline": False,                         # set in enable(): True when not in tmux
    "can_display": True,                     # False -> no usable target; skip + warn
    "made_pane": None,                       # pane id enable() auto-split, if any
}

_cache = os.path.expanduser(_env("CACHE", "~/.cache/plotty"))
os.makedirs(_cache, exist_ok=True)
_last = os.path.join(_cache, "last.png")
_pidfile = os.path.join(_cache, "viewer.pid")
_config = os.path.join(_cache, "config.json")
_histdir = os.path.join(_cache, "hist")

_warned = set()


def _warn_once(key, msg):
    """Print a warning to stderr once per topic (avoids per-cell noise in the
    IPython hook when something is persistently broken)."""
    if key not in _warned:
        _warned.add(key)
        print(f"[{__name__}] {msg}", file=sys.stderr)


def _write_config():
    """Publish the live display settings for the viewer.

    The viewer re-reads this on every draw, so re-running enable(size=...,
    bg=..., imgcat=...) takes effect on the next figure without a restart."""
    data = {"imgcat": _cfg["imgcat"], "size": _cfg["size"],
            "clear": _cfg["clear"], "bg": _cfg["bg"]}
    tmp = _config + ".part"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, _config)
    except OSError:
        pass


def _load_settings():
    """Viewer/--render side: env vars bootstrap, then config.json (live
    updates from the backend) takes precedence."""
    _cfg["imgcat"] = _env("IMGCAT", "") or None      # empty -> built-in encoder
    _cfg["size"] = _env("SIZE", _cfg["size"])
    _cfg["clear"] = _env("CLEAR", "1" if _cfg["clear"] else "0") != "0"
    _cfg["bg"] = _env("BG", _cfg.get("bg"))
    try:
        with open(_config) as f:
            data = json.load(f)
        for key in ("imgcat", "size", "clear", "bg"):
            if key in data:
                _cfg[key] = data[key]
    except (OSError, ValueError):
        pass


# ---- renderer detection -----------------------------------------------------

def _is_sixel(cmd):
    return bool(cmd) and "sixel" in cmd.lower()


def _auto_imgcat():
    """Return the first renderer command available on PATH, else None."""
    for cmd in _CANDIDATES:
        if shutil.which(shlex.split(cmd)[0]):
            return cmd
    return None


def _renderer_for(name):
    """The candidate template whose program is exactly `name`, or None."""
    for cmd in _CANDIDATES:
        if shlex.split(cmd)[0] == name:
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


def _parse_da1(resp):
    """Parse a DA1 reply (ESC[?<attrs>c): True if attribute 4 (sixel) is
    advertised, False if a reply lacks it, None if there is no parseable reply."""
    m = re.search(rb"\[\?([\d;]*)c", resp)
    if not m:
        return None
    return b"4" in m.group(1).split(b";")


def _stdout_supports_sixel():
    """Best-effort terminal sixel check via a DA1 query.

    Returns False when stdout is not a terminal or the terminal's DA1 reply
    lacks the sixel attribute (IDE consoles answer DA1 without it — dumping
    raw sixel there just prints escape garbage). None means undeterminable.
    """
    try:
        if not sys.stdout.isatty():
            return False
        if not sys.stdin.isatty():
            return None                          # can't query without the tty
        import termios
        import tty as _tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        resp = b""
        try:
            _tty.setcbreak(fd)
            sys.stdout.write("\x1b[c")           # DA1: "what are you?"
            sys.stdout.flush()
            while select.select([fd], [], [], 0.3)[0]:
                resp += os.read(fd, 64)
                if resp.rstrip().endswith(b"c"):
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return _parse_da1(resp)
    except Exception:
        return None


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


def _parse_bg(value):
    """'#rrggbb' (or 'rrggbb') -> (r, g, b); None/invalid -> white."""
    if not value:
        return (255, 255, 255)
    v = str(value).lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{6}", v):
        return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))
    _warn_once("bg", f"ignoring invalid bg {value!r} (use '#rrggbb')")
    return (255, 255, 255)


def _load_rgb(path):
    """Read a PNG into an (H, W, 3) uint8 array, compositing alpha over the
    configured background color (white by default; PLOTTY_BG / enable(bg=...)
    for dark terminals)."""
    a = mpimg.imread(path)                       # matplotlib reads PNG w/o Pillow
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if np.issubdtype(a.dtype, np.floating):
        a = (a * 255.0).round().clip(0, 255).astype(np.uint8)
    else:
        a = a.astype(np.uint8)
    if a.shape[2] == 4:
        bg = np.array(_parse_bg(_cfg.get("bg")), np.float32)
        alpha = a[..., 3:4].astype(np.float32) / 255.0
        rgb = a[..., :3].astype(np.float32)
        a = (rgb * alpha + bg * (1.0 - alpha)).round().astype(np.uint8)
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
    """Quantize to <=ncolors. Returns (palette (K,3) uint8, indices (N,) int).

    Operates on the image's *distinct* colors (packed RGB ints) instead of raw
    pixels: an exact, lossless palette when there are <=ncolors distinct colors,
    else median-cut over the distinct colors weighted by pixel counts. Distinct
    colors number in the hundreds even for anti-aliased plots (vs ~10^5 pixels),
    so both paths are fast, and the weighted palette is identical to running
    median-cut over the raw pixels.
    """
    pixels = rgb.reshape(-1, 3).astype(np.int32)
    packed = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
    uniq, inverse, counts = np.unique(packed, return_inverse=True,
                                      return_counts=True)
    inverse = inverse.reshape(-1).astype(np.int32)
    colors = np.stack([(uniq >> 16) & 0xFF, (uniq >> 8) & 0xFF, uniq & 0xFF],
                      axis=1).astype(np.uint8)
    if colors.shape[0] <= ncolors:
        return colors, inverse                   # exact palette, lossless
    boxes = [_make_box(colors, np.arange(colors.shape[0]))]
    while len(boxes) < ncolors:
        si, best = -1, 0
        for i, (ids, rng, _) in enumerate(boxes):
            if ids.size > 1 and rng > best:
                si, best = i, rng
        if si < 0 or best == 0:
            break                                # all boxes are single-colour
        ids, _, ch = boxes.pop(si)
        ids = ids[np.argsort(colors[ids, ch], kind="stable")]
        cum = np.cumsum(counts[ids])             # split at the weighted median
        mid = int(np.searchsorted(cum, cum[-1] / 2)) + 1
        mid = min(max(mid, 1), ids.size - 1)
        boxes.append(_make_box(colors, ids[:mid]))
        boxes.append(_make_box(colors, ids[mid:]))
    palette = np.empty((len(boxes), 3), np.uint8)
    color_pal = np.empty(colors.shape[0], np.int32)
    for i, (ids, _, _) in enumerate(boxes):
        w = counts[ids].astype(np.float64)[:, None]
        palette[i] = np.round((colors[ids] * w).sum(axis=0) / w.sum())
        color_pal[ids] = i
    return palette, color_pal[inverse]


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
    """Negative ints index the current window's panes Python-style (-1 = last).

    The REPL's own pane ($TMUX_PANE) is excluded when other panes exist —
    drawing into the pane the user is typing in is never what they want.
    """
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
        own = os.environ.get("TMUX_PANE")
        if len(ids) > 1 and own in ids:
            ids.remove(own)
        if ids:
            return ids[max(idx, -len(ids))]    # stable pane id (%N)
    except OSError:
        pass
    return str(target)


def _ensure_separate_pane(target_pane, verbose):
    """Resolve the target pane, creating one if the REPL's pane is the only one.

    send-keys / sixel into the pane the user is typing in just injects garbage
    into their console — if there is no separate pane, split one off; if that
    fails, disable display and say so instead of typing into the REPL.
    """
    pane = _resolve_pane(target_pane)
    own = os.environ.get("TMUX_PANE")
    if not own or pane != own:
        return pane
    try:
        out = subprocess.run([_cfg["tmux"], "split-window", "-d", "-h", "-t", own,
                              "-P", "-F", "#{pane_id}"],
                             capture_output=True, text=True, check=False)
        new = out.stdout.strip()
    except OSError:
        new = ""
    if new:
        _cfg["made_pane"] = new                  # remember: disable(close_pane=True)
        if verbose:
            print(f"[{__name__}] no separate pane to draw into: created plot "
                  f"pane {new} (tmux split-window)", file=sys.stderr)
        return new
    _cfg["can_display"] = False
    if verbose:
        print(f"[{__name__}] cannot display figures: this tmux window has no "
              f"separate pane and creating one failed — split a pane "
              f"(prefix+\") and re-run enable()", file=sys.stderr)
    return pane


# ---- talking to the viewer (or send-keys fallback) --------------------------

def _read_pid():
    try:
        with open(_pidfile) as f:
            return int(f.readline().strip())
    except (OSError, ValueError):
        return None


def _read_viewer_pane():
    """The tmux pane the viewer runs in (line 2 of the pidfile), or None."""
    try:
        with open(_pidfile) as f:
            f.readline()
            return f.readline().strip() or None
    except OSError:
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


def _hist_files():
    """History snapshots, newest first (index 0 == the current figure)."""
    try:
        names = sorted(os.listdir(_histdir), reverse=True)
    except OSError:
        return []
    return [os.path.join(_histdir, n) for n in names if n.endswith(".png")]


def _record_history():
    """Snapshot the just-published last.png into the history ring and prune.

    Uses a hardlink (free): os.replace on the next publish unlinks last.png's
    name from the old inode, so the snapshot keeps the old bytes alive."""
    try:
        keep = int(_cfg.get("hist") or 0)
    except (TypeError, ValueError):
        keep = 10
    if keep <= 0:
        return
    try:
        os.makedirs(_histdir, exist_ok=True)
        name = os.path.join(_histdir, f"fig-{time.time_ns():020d}.png")
        try:
            os.link(_last, name)
        except OSError:
            shutil.copyfile(_last, name)
        for old in _hist_files()[keep:]:
            try:
                os.remove(old)
            except OSError:
                pass
    except OSError:
        pass


def _publish(fig):
    """Save the figure once, straight to last.png via temp-file + os.replace
    (atomic hand-off: the viewer never sees a partial file). True on success."""
    kw = {"format": "png", "bbox_inches": "tight"}
    if _cfg["dpi"]:
        try:
            kw["dpi"] = float(_cfg["dpi"])
        except (TypeError, ValueError):
            _warn_once("dpi", f"ignoring invalid dpi {_cfg['dpi']!r}")
    tmp = _last + ".part"
    try:
        fig.savefig(tmp, **kw)
        os.replace(tmp, _last)
    except OSError as exc:
        _warn_once("publish", f"cannot write {_last}: {exc}")
        return False
    _record_history()
    return True


def _display_figure(fig):
    if not _cfg.get("can_display", True):
        _warn_once("display", "figure not displayed — no usable display target "
                              "(see the enable() warnings); fix it and re-run "
                              "enable()")
        return
    if not _publish(fig):
        return
    if _cfg["inline"]:
        _write_inline(_last)
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


def show(fig=None, *args, **kwargs):
    """Display pyplot-managed figures, or an explicit Figure passed as `fig`.

    Passing a figure covers non-pyplot figures (matplotlib.figure.Figure built
    directly), which never register with pyplot and would otherwise not show.
    Such figures are not auto-closed."""
    if fig is not None:
        _display_figure(fig)
        return
    managers = Gcf.get_all_fig_managers()
    if not managers:
        return
    for manager in managers:
        _display_figure(manager.canvas.figure)
    if _cfg["close"]:
        Gcf.destroy_all()


def save(path):
    """Copy the most recently displayed figure (full-resolution PNG) to path."""
    if not os.path.exists(_last):
        raise FileNotFoundError("plotty has not displayed a figure yet")
    dst = os.path.expanduser(path)
    shutil.copyfile(_last, dst)
    return dst


def redraw():
    if not _cfg.get("can_display", True):
        return
    if _cfg["inline"]:
        if os.path.exists(_last):
            _write_inline(_last)
    elif not _signal_viewer() and os.path.exists(_last):
        _emit()


# ---- the viewer (runs in the plot pane) -------------------------------------

def view():
    """Viewer loop: redraw on SIGUSR1 (new figure) / SIGWINCH (resize), exit on
    SIGTERM/SIGINT/SIGHUP. When the pane tty is interactive, single keys
    navigate figure history: p/k = older, n/j = newer, q = quit.

    Signal handlers do no work — the kernel writes the signal number to a
    self-pipe (signal.set_wakeup_fd) and all rendering/writing happens in the
    main loop. That avoids handler reentrancy during resize bursts and
    unguarded writes to a dying pty, and every exit path is a clean
    os._exit(0): an abnormal viewer exit (traceback or fatal signal) makes
    macOS pop a "Python quit unexpectedly" crash dialog when the surrounding
    session is torn down. Draining the pipe for ~50 ms also coalesces signal
    bursts, so a resize storm redraws once. Display settings are re-read from
    config.json on every draw, so enable(...) changes apply live.
    """
    _load_settings()

    draw_sigs = {int(getattr(signal, n)) for n in ("SIGUSR1", "SIGWINCH")
                 if hasattr(signal, n)}
    quit_sigs = {int(getattr(signal, n)) for n in ("SIGTERM", "SIGINT", "SIGHUP")
                 if hasattr(signal, n)}
    usr1 = int(getattr(signal, "SIGUSR1", -1))
    kb_fd = kb_old = None
    offset = 0                                   # 0 = live; k > 0 = k figures back

    def _cleanup():
        if kb_old is not None:
            try:
                import termios
                termios.tcsetattr(kb_fd, termios.TCSADRAIN, kb_old)
            except Exception:
                pass
        try:
            if _read_pid() == os.getpid():
                os.remove(_pidfile)
        except OSError:
            pass

    def _current():
        """(path, status note) for the figure the viewer should show."""
        if offset <= 0:
            return _last, ""
        files = _hist_files()
        if offset >= len(files):
            return _last, ""
        return files[offset], f"[{offset}/{len(files) - 1}] n=newer p=older q=quit"

    def _draw(path, note=""):
        """Render `path` to stdout. False means the tty is gone: exit."""
        if not path or not os.path.exists(path):
            return True
        _load_settings()                         # live size/bg/clear/renderer
        try:
            data = _render_bytes(path, _out_fd())
        except Exception:
            return True                          # bad/partial image: skip frame
        try:
            out = sys.stdout.buffer
            if _cfg["clear"]:
                out.write(b"\x1b[H\x1b[2J")      # home + clear screen
            out.write(data)
            if note:
                out.write(b"\r\n" + note.encode())
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
        try:
            if sys.stdin.isatty():               # single-key history navigation
                import termios
                import tty as _tty
                kb_fd = sys.stdin.fileno()
                kb_old = termios.tcgetattr(kb_fd)
                _tty.setcbreak(kb_fd)
        except Exception:
            kb_fd = kb_old = None
        tmp = _pidfile + ".part"
        with open(tmp, "w") as f:                # pid + the pane we live in
            f.write(f"{os.getpid()}\n{os.environ.get('TMUX_PANE', '')}\n")
        os.replace(tmp, _pidfile)                # atomic, like last.png
        fds = [rfd] + ([kb_fd] if kb_fd is not None else [])
        ok = _draw(*_current())
        while ok:
            ready = select.select(fds, [], [])[0]    # blocks: zero CPU idle
            sigs, keys = set(), b""
            if rfd in ready:
                sigs = set(os.read(rfd, 128))
                while select.select([rfd], [], [], 0.05)[0]:
                    sigs |= set(os.read(rfd, 128))   # coalesce signal bursts
            if kb_fd is not None and kb_fd in ready:
                keys = os.read(kb_fd, 16)
            if (sigs & quit_sigs) or b"q" in keys:
                break
            redraw = bool(sigs & draw_sigs)
            if usr1 in sigs:
                offset = 0                       # new figure: jump back to live
            for key in keys:
                if key in (ord("p"), ord("k")):      # older
                    offset = min(offset + 1, max(len(_hist_files()) - 1, 0))
                    redraw = True
                elif key in (ord("n"), ord("j")):    # newer
                    offset = max(offset - 1, 0)
                    redraw = True
            if redraw:
                ok = _draw(*_current())
    except BaseException:
        pass                                     # never crash out of the viewer
    _cleanup()
    os._exit(0)


# ---- setup / teardown -------------------------------------------------------

def _ensure_viewer():
    pid = _read_pid()
    if _alive(pid) and _is_viewer(pid):      # don't trust recycled pids
        vpane = _read_viewer_pane()
        if not vpane or vpane == str(_cfg["pane"]):
            return                           # already in the right pane
        try:
            os.kill(pid, signal.SIGTERM)     # target moved: restart it there
        except OSError:
            pass
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

    The default (imgcat=None, no PLOTTY_IMGCAT) is the built-in, dependency-free
    encoder. "chafa"/"img2sixel"/"magick"/"convert" select that external tool
    (warning + built-in fallback if it isn't installed), "auto" picks the first
    external tool found on PATH, "" / "builtin" / False force the built-in, and
    any other string is used as a custom command.
    """
    if imgcat is None:
        imgcat = _env("IMGCAT", None)
    if imgcat in (None, "", "builtin", False):
        return None                              # built-in encoder (the default)
    if imgcat == "auto":
        imgcat = _auto_imgcat()
        if imgcat is None:
            if verbose:
                print(f"[{__name__}] no external renderer on PATH; using the "
                      f"built-in sixel encoder", file=sys.stderr)
            return None
    elif _renderer_for(imgcat):                  # bare tool name, e.g. "chafa"
        if shutil.which(imgcat):
            imgcat = _renderer_for(imgcat)
        else:
            if verbose:
                print(f"[{__name__}] {imgcat} not found on PATH; using the "
                      f"built-in sixel encoder", file=sys.stderr)
            return None
    if verbose and not _is_sixel(imgcat):
        print(f"[{__name__}] {shlex.split(imgcat)[0]} is not sixel, so image "
              f"display may not work over ssh", file=sys.stderr)
    return imgcat


def enable(target_pane=-1, imgcat=None, clear=True, tmux="tmux", dpi=None,
           close=True, size=None, bg=None, hist=None, inline=None, viewer=True,
           verbose=1):
    """Activate plotty: detect a renderer, point at a pane, start the viewer.

    inline=None (default) auto-selects: inline mode when not in tmux, viewer-pane
    mode when in tmux. In inline mode the backend renders sixel itself (no viewer
    process) and writes it to the target pane's tty when in tmux, or to this
    terminal's stdout when not. inline=True forces inline even inside tmux;
    inline=False forces viewer-pane mode. `PLOTTY_INLINE=1/0` sets the default.

    imgcat=None (default) uses the built-in, dependency-free sixel encoder.
    Pass imgcat="chafa" / "img2sixel" / "magick" to use that external tool
    (slightly faster, better resampling; falls back to built-in with a warning
    if it isn't installed), imgcat="auto" to pick the first external tool on
    PATH, or a full command string to use it verbatim. `PLOTTY_IMGCAT` sets the
    default. This applies to both viewer and inline modes.

    size (display width in cells, default 60) and dpi (matplotlib savefig DPI;
    None = matplotlib's own default) control display size and source-image
    resolution respectively. Both fall back to `PLOTTY_SIZE` / `PLOTTY_DPI` when
    the argument is None. Raise dpi when displaying at a large size so the source
    PNG has enough pixels to stay sharp (else the renderer upscales it).

    bg ('#rrggbb', default white; `PLOTTY_BG`) is the background that transparent
    figure regions are composited over by the built-in encoder — set it to your
    terminal's background for dark setups. hist (default 10; `PLOTTY_HIST`) is
    how many recent figures are kept for the viewer's history keys (p/n).

    Settings are also published to the cache config, so re-running enable()
    with new values updates a running viewer live (a changed target_pane
    restarts it in the new pane).
    """
    _cfg["tmux"] = tmux
    _cfg["clear"] = clear
    _cfg["dpi"] = _env("DPI", None) if dpi is None else dpi
    _cfg["close"] = close
    _cfg["size"] = _env("SIZE", 60) if size is None else size
    _cfg["bg"] = _env("BG", None) if bg is None else bg
    _cfg["hist"] = _env("HIST", "10") if hist is None else hist
    _cfg["made_pane"] = None

    _cfg["imgcat"] = _resolve_imgcat(imgcat, verbose)
    _write_config()                              # viewer re-reads this per draw

    matplotlib.use(f"module://{__name__}")
    matplotlib.interactive(True)

    _cfg["inline"] = _resolve_inline(inline)
    _cfg["can_display"] = True
    intmux = os.environ.get("TMUX") is not None
    if not intmux and not _cfg["inline"]:
        if verbose:
            print(f"[{__name__}] not inside tmux: viewer-pane mode is unavailable, "
                  f"falling back to inline display", file=sys.stderr)
        _cfg["inline"] = True
    _health_check(verbose)
    if _cfg["inline"]:
        if intmux:
            # pipe sixel to a separate pane's tty (never the REPL's own pane)
            _cfg["pane"] = _ensure_separate_pane(target_pane, verbose)
        elif inline is not True:                 # auto-selected: verify the terminal
            if _stdout_supports_sixel() is False:
                _cfg["can_display"] = False
                if verbose:
                    print(f"[{__name__}] this terminal does not appear to support "
                          f"sixel — figures will not be displayed (use a "
                          f"sixel-capable terminal or tmux; enable(inline=True) "
                          f"forces output anyway)", file=sys.stderr)
    else:
        _cfg["pane"] = _ensure_separate_pane(target_pane, verbose)
        if _cfg["can_display"] and viewer:
            _ensure_viewer()
    hook()


def disable(close_pane=False, verbose=1):
    """Stop the viewer, unhook auto-display, and quiet matplotlib output.

    close_pane=True also closes the plot pane — but only if enable() created
    it via auto-split (a pane the user made themselves is never touched)."""
    pid = _read_pid()
    if _alive(pid) and _is_viewer(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    if close_pane and _cfg.get("made_pane"):
        subprocess.run([_cfg["tmux"], "kill-pane", "-t", _cfg["made_pane"]],
                       capture_output=True, check=False)
        _cfg["made_pane"] = None
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
    if verbose:
        print(f"[{__name__}] disabled — figures will no longer be displayed "
              f"(matplotlib backend: agg)", file=sys.stderr)


def status():
    """Print a one-call diagnostic summary of plotty's current state."""
    intmux = os.environ.get("TMUX") is not None
    if not _cfg.get("can_display", True):
        mode = "DISABLED (no usable display target — see enable() warnings)"
    elif _cfg["inline"]:
        target = f"tmux pane {_cfg['pane']}" if intmux else "stdout"
        mode = f"inline -> {target}"
    else:
        mode = f"viewer pane {_cfg['pane']}"
    pid = _read_pid()
    if _alive(pid) and _is_viewer(pid):
        viewer = f"running (pid {pid}, pane {_read_viewer_pane() or '?'})"
    else:
        viewer = "not running"
    try:
        st = os.stat(_last)
        last = time.strftime("%H:%M:%S", time.localtime(st.st_mtime)) \
            + f" ({st.st_size} bytes)"
    except OSError:
        last = "never"
    lines = [
        f"plotty {__version__}",
        f"  mode:      {mode}",
        f"  renderer:  {_cfg['imgcat'] or 'built-in sixel encoder'}",
        f"  size:      {_cfg['size']} cells   dpi: {_cfg['dpi'] or 'default'}   "
        f"bg: {_cfg.get('bg') or 'white'}",
        f"  viewer:    {viewer}",
        f"  last fig:  {last}   history: {len(_hist_files())} kept",
        f"  cache:     {_cache}",
    ]
    if intmux:
        ver = _tmux_version()
        feats = _tmux_features() or ""
        lines.append(f"  tmux:      {'.'.join(map(str, ver)) if ver else '?'}   "
                     f"sixel feature: {'yes' if 'sixel' in feats else 'not reported'}")
    print("\n".join(lines))


if __name__ == "__main__":
    if "--view" in sys.argv:
        view()
    elif "--render" in sys.argv:
        # render last.png to this pane's stdout (send-keys fallback)
        _load_settings()
        if os.path.exists(_last):
            sys.stdout.buffer.write(_render_bytes(_last, _out_fd()))
            sys.stdout.buffer.flush()
