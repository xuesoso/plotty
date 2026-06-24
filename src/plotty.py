"""
plotty - display matplotlib figures in a dedicated tmux pane (plot + tty).

The Python/Jupyter analogue of MuxDisplay.jl, built for SSH + tmux. The backend
(this module) runs in your REPL; a tiny viewer runs in the plot pane and redraws
on SIGUSR1 (new figure) and SIGWINCH (pane resize/zoom). Only the rendered sixel
bytes cross SSH, so it works the same locally and over a remote session.

    import plotty
    plotty.enable()                      # auto-detects renderer + plot pane
    plotty.enable(target_pane=2)         # or pick a pane explicitly
    plotty.disable()                     # stop the viewer + auto-display

Rendering is zero-dependency and protocol-aware by default: plotty detects
whether the terminal supports sixel and uses the built-in sixel encoder if so,
else the built-in kitty-graphics encoder (Unicode placeholders — for ghostty/
kitty; inside tmux it requires `tmux set -g allow-passthrough on`, single tmux
layer only). Override with enable(imgcat=...): "builtin" forces sixel, "kitty"
forces kitty graphics, "chafa"/"img2sixel"/"magick" use that external sixel
tool (slightly faster, better resampling), or pass a full custom command. A
non-sixel custom command warns that it may not work over SSH.

Display modes: a viewer process running in a tmux pane (default in tmux), or
"inline" mode which renders sixel itself (no viewer) and writes it to the target
pane's tty when in tmux, or to the current terminal's stdout when not. Choose
with enable(inline=...) / PLOTTY_INLINE.

plotty never draws into the pane you are typing in: by default it keeps a
dedicated plot pane in the current window, splitting one off the first time and
reusing it afterwards (and recreating it if you close it) — so it never clobbers
an editor or any pane you opened yourself. Pass target_pane=... to aim at a
specific pane instead. Without a usable sixel display (e.g. an IDE console), it
warns instead of printing escape garbage.

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

# tmux pane option that tags the dedicated plot pane enable() splits off, so it
# can be found and reused on later enable() calls (this process or another). The
# tag dies with the pane, so "user killed it" is just "the tag is gone".
_DEDICATED_OPT = "@plotty"

# External sixel renderer candidates (opt-in: a bare tool name like
# imgcat="chafa" selects its template). All sixel — the only non-sixel path is
# the built-in kitty-graphics encoder (imgcat="kitty"). Placeholders are
# substituted at render time:
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
    "auto_pane": False,                      # True -> dedicated pane, recreate if killed
    "verbose": 1,                            # enable()'s verbosity, reused on recreate
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


_term_probe = None                               # cached (sixel, kitty) result


def _probe_terminal():
    """Query the terminal once for graphics support: (sixel, kitty), each
    True/False, or None when undeterminable. Cached for the process.

    One round-trip: a kitty-graphics query (a=q; silently ignored by other
    terminals) followed by DA1. Sixel = attribute 4 in the DA1 reply; kitty =
    a graphics APC response arriving before it. False when stdout is not a
    terminal (dumping escape bytes into a pipe/IDE console is never wanted).
    """
    global _term_probe
    if _term_probe is not None:
        return _term_probe
    sixel = kitty = None
    try:
        if not sys.stdout.isatty():
            sixel = kitty = False
        elif sys.stdin.isatty():                 # can't query without the tty
            import termios
            import tty as _tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            resp = b""
            try:
                _tty.setcbreak(fd)
                sys.stdout.write("\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\"
                                 "\x1b[c")
                sys.stdout.flush()
                while select.select([fd], [], [], 0.3)[0]:
                    resp += os.read(fd, 256)
                    if resp.rstrip().endswith(b"c"):
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sixel = _parse_da1(resp)
            kitty = (b"\x1b_G" in resp) if resp else None
    except Exception:
        pass
    _term_probe = (sixel, kitty)
    return _term_probe


def _stdout_supports_sixel():
    """Best-effort DA1 sixel check (see _probe_terminal); None = unknown."""
    return _probe_terminal()[0]


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


# ---- kitty graphics encoder (Unicode placeholders; for ghostty/kitty) -------
#
# Opt-in via enable(imgcat="kitty"). Transmits the PNG with the kitty graphics
# protocol as a *virtual* placement (U=1) and then draws it as plain Unicode
# placeholder text (U+10EEEE cells with row/column diacritics, image id in the
# foreground color). Because the placeholders are ordinary text, tmux tracks
# them in its grid and redraws them itself — the image survives pane resize,
# zoom and pane switches. This is the one robust non-sixel path inside tmux;
# ghostty (which refuses sixel) and kitty are the terminals that support it.
# Inside tmux the transmission APCs must be passthrough-wrapped, which requires
# `set -g allow-passthrough on` (tmux >= 3.3); the placeholder text is not
# wrapped. Nested tmux is not supported (passthrough does not survive two
# layers). q=2 keeps the terminal quiet: replies cannot route back through tmux.

_PLACEHOLDER = "\U0010EEEE"
_DIACRITICS = (                                  # kitty's rowcolumn-diacritics
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346,
    0x034A, 0x034B, 0x034C, 0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363,
    0x0364, 0x0365, 0x0366, 0x0367, 0x0368, 0x0369, 0x036A, 0x036B, 0x036C,
    0x036D, 0x036E, 0x036F, 0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592,
    0x0593, 0x0594, 0x0595, 0x0597, 0x0598, 0x0599, 0x059C, 0x059D, 0x059E,
    0x059F, 0x05A0, 0x05A1, 0x05A8, 0x05A9, 0x05AB, 0x05AC, 0x05AF, 0x05C4,
    0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615, 0x0616, 0x0617, 0x0657,
    0x0658, 0x0659, 0x065A, 0x065B, 0x065D, 0x065E, 0x06D6, 0x06D7, 0x06D8,
    0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2, 0x06E4,
    0x06E7, 0x06E8, 0x06EB, 0x06EC, 0x0730, 0x0732, 0x0733, 0x0735, 0x0736,
    0x073A, 0x073D, 0x073F, 0x0740, 0x0741, 0x0743, 0x0745, 0x0747, 0x0749,
    0x074A, 0x07EB, 0x07EC, 0x07ED, 0x07EE, 0x07EF, 0x07F0, 0x07F1, 0x07F3,
    0x0816, 0x0817, 0x0818, 0x0819, 0x081B, 0x081C, 0x081D, 0x081E, 0x081F,
    0x0820, 0x0821, 0x0822, 0x0823, 0x0825, 0x0826, 0x0827, 0x0829, 0x082A,
    0x082B, 0x082C, 0x082D, 0x0951, 0x0953, 0x0954, 0x0F82, 0x0F83, 0x0F86,
    0x0F87, 0x135D, 0x135E, 0x135F, 0x17DD, 0x193A, 0x1A17, 0x1A75, 0x1A76,
    0x1A77, 0x1A78, 0x1A79, 0x1A7A, 0x1A7B, 0x1A7C, 0x1B6B, 0x1B6D, 0x1B6E,
    0x1B6F, 0x1B70, 0x1B71, 0x1B72, 0x1B73, 0x1CD0, 0x1CD1, 0x1CD2, 0x1CDA,
    0x1CDB, 0x1CE0, 0x1DC0, 0x1DC1, 0x1DC3, 0x1DC4, 0x1DC5, 0x1DC6, 0x1DC7,
    0x1DC8, 0x1DC9, 0x1DCB, 0x1DCC, 0x1DD1, 0x1DD2, 0x1DD3, 0x1DD4, 0x1DD5,
    0x1DD6, 0x1DD7, 0x1DD8, 0x1DD9, 0x1DDA, 0x1DDB, 0x1DDC, 0x1DDD, 0x1DDE,
    0x1DDF, 0x1DE0, 0x1DE1, 0x1DE2, 0x1DE3, 0x1DE4, 0x1DE5, 0x1DE6, 0x1DFE,
    0x20D0, 0x20D1, 0x20D4, 0x20D5, 0x20D6, 0x20D7, 0x20DB, 0x20DC, 0x20E1,
    0x20E7, 0x20E9, 0x20F0, 0x2CEF, 0x2CF0, 0x2CF1, 0x2DE0, 0x2DE1, 0x2DE2,
    0x2DE3, 0x2DE4, 0x2DE5, 0x2DE6, 0x2DE7, 0x2DE8, 0x2DE9, 0x2DEA, 0x2DEB,
    0x2DEC, 0x2DED, 0x2DEE, 0x2DEF, 0x2DF0, 0x2DF1, 0x2DF2, 0x2DF3, 0x2DF4,
    0x2DF5, 0x2DF6, 0x2DF7, 0x2DF8, 0x2DF9, 0x2DFA, 0x2DFB, 0x2DFC, 0x2DFD,
    0x2DFE, 0x2DFF, 0xA66F, 0xA67C, 0xA67D, 0xA6F0, 0xA6F1, 0xA8E0, 0xA8E1,
    0xA8E2, 0xA8E3, 0xA8E4, 0xA8E5, 0xA8E6, 0xA8E7, 0xA8E8, 0xA8E9, 0xA8EA,
    0xA8EB, 0xA8EC, 0xA8ED, 0xA8EE, 0xA8EF, 0xA8F0, 0xA8F1, 0xAAB0, 0xAAB2,
    0xAAB3, 0xAAB7, 0xAAB8, 0xAABE, 0xAABF, 0xAAC1, 0xFE20, 0xFE21, 0xFE22,
    0xFE23, 0xFE24, 0xFE25, 0xFE26, 0x10A0F, 0x10A38, 0x1D185, 0x1D186,
    0x1D187, 0x1D188, 0x1D189, 0x1D1AA, 0x1D1AB, 0x1D1AC, 0x1D1AD, 0x1D242,
    0x1D243, 0x1D244,
)


def _kitty_id():
    """Image id (1-255, fits the 8-bit SGR fg encoding), stable per process."""
    return (os.getpid() % 255) + 1


def _wrap_tmux(seq):
    """Wrap an escape sequence in tmux's passthrough envelope (ESCs doubled)."""
    return b"\x1bPtmux;" + seq.replace(b"\x1b", b"\x1b\x1b") + b"\x1b\\"


def _png_size(data):
    """(width, height) from a PNG IHDR header, or None."""
    import struct
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def _kitty_bytes(png, cols, rows, wrap):
    """Kitty-graphics stream: delete the old image, transmit `png` as a virtual
    placement of cols x rows cells, then print the Unicode placeholder grid."""
    import base64
    iid = _kitty_id()
    apcs = [b"\x1b_Ga=d,d=I,i=%d,q=2\x1b\\" % iid]      # free the previous image
    payload = base64.standard_b64encode(png)
    chunks = [payload[i:i + 4096] for i in range(0, len(payload), 4096)] or [b""]
    head = b"a=T,U=1,q=2,f=100,t=d,i=%d,c=%d,r=%d" % (iid, cols, rows)
    for n, chunk in enumerate(chunks):
        more = 1 if n < len(chunks) - 1 else 0
        keys = (head + b",m=%d" % more) if n == 0 else b"m=%d" % more
        apcs.append(b"\x1b_G" + keys + b";" + chunk + b"\x1b\\")
    out = bytearray()
    for apc in apcs:
        out += _wrap_tmux(apc) if wrap else apc
    out += b"\x1b[38;5;%dm" % iid                       # id rides the fg color
    for row in range(rows):
        line = "".join(_PLACEHOLDER + chr(_DIACRITICS[row]) + chr(_DIACRITICS[col])
                       for col in range(cols))
        out += line.encode("utf-8")
        if row < rows - 1:
            out += b"\r\n"
    out += b"\x1b[39m\r\n"
    return bytes(out)


def _render_kitty(path, fd):
    """Render `path` via kitty graphics + Unicode placeholders (imgcat="kitty")."""
    with open(path, "rb") as f:
        png = f.read()
    if _cfg.get("bg"):                                  # honor bg compositing
        import io
        buf = io.BytesIO()
        mpimg.imsave(buf, _load_rgb(path), format="png")
        png = buf.getvalue()
    iw, ih = _png_size(png) or (4, 3)
    cols, rows, xpix, ypix = _winsize(fd)
    cell_w = (xpix / cols) if (xpix and cols) else 10.0
    cell_h = (ypix / rows) if (ypix and rows) else 20.0
    limit = len(_DIACRITICS)
    c = max(1, min(int(_cfg["size"]), cols or limit, limit))
    r = max(1, round(c * cell_w * ih / (iw * cell_h)))
    max_r = min(max((rows or 24) - 1, 1), limit)
    if r > max_r:                                       # too tall: keep aspect
        c = max(1, round(c * max_r / r))
        r = max_r
    return _kitty_bytes(png, c, r, wrap=os.environ.get("TMUX") is not None)


def _render_bytes(path, fd):
    """Return the terminal byte stream to display `path` (external cmd or built-in)."""
    cmd = _cfg["imgcat"]
    if cmd == "kitty":
        return _render_kitty(path, fd)
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


def _split_plot_pane(own):
    """Split a detached plot pane next to `own` (or the current pane if `own` is
    empty); return its tmux pane id, or "" on failure. `-d` keeps focus in the
    REPL; `-h` splits horizontally (side by side)."""
    cmd = [_cfg["tmux"], "split-window", "-d", "-h"]
    if own:
        cmd += ["-t", own]
    cmd += ["-P", "-F", "#{pane_id}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return out.stdout.strip()
    except OSError:
        return ""


def _find_dedicated_pane():
    """The current window's pane tagged as plotty's dedicated plot pane (the
    `@plotty` pane option), or None — for which "missing" means it was never
    created or the user closed it (the tag is destroyed with the pane)."""
    try:
        out = subprocess.run(
            [_cfg["tmux"], "list-panes", "-F", "#{pane_id} #{%s}" % _DEDICATED_OPT],
            capture_output=True, text=True, check=False)
    except OSError:
        return None
    for line in out.stdout.splitlines():
        pane_id, _, tag = line.partition(" ")
        if tag == "1" and pane_id:
            return pane_id
    return None


def _ensure_dedicated_pane(verbose):
    """Resolve plotty's own plot pane (the target_pane="auto" default).

    Reuse the dedicated pane if it's still alive; otherwise split a fresh one and
    tag it `@plotty` so the next enable() finds it again. Unlike picking the last
    pane, this never draws into a pane the user opened themselves (an editor,
    logs, …); if they close the dedicated pane, a new one is made next time. As
    in _ensure_separate_pane, a failed split disables display with a warning
    rather than typing into the REPL.
    """
    own = os.environ.get("TMUX_PANE")
    pane = _find_dedicated_pane()
    if pane and pane != own:
        _cfg["made_pane"] = pane                  # plotty owns it: disable(close_pane=True)
        return pane
    new = _split_plot_pane(own)
    if new:
        subprocess.run([_cfg["tmux"], "set", "-p", "-t", new, _DEDICATED_OPT, "1"],
                       capture_output=True, check=False)
        _cfg["made_pane"] = new
        if verbose:
            print(f"[{__name__}] created dedicated plot pane {new} "
                  f"(tmux split-window)", file=sys.stderr)
        return new
    _cfg["can_display"] = False
    if verbose:
        print(f"[{__name__}] cannot display figures: creating a dedicated plot "
              f"pane failed — split a pane (prefix+\") and re-run enable()",
              file=sys.stderr)
    return pane or own or str(_cfg["pane"])


def _ensure_separate_pane(target_pane, verbose):
    """Resolve an explicit target pane, creating one if the REPL's pane is the
    only one.

    send-keys / sixel into the pane the user is typing in just injects garbage
    into their console — if there is no separate pane, split one off; if that
    fails, disable display and say so instead of typing into the REPL.
    """
    pane = _resolve_pane(target_pane)
    own = os.environ.get("TMUX_PANE")
    if not own or pane != own:
        return pane
    new = _split_plot_pane(own)
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


def _resolve_display_pane(target_pane, verbose):
    """Pick the plot pane. The default (target_pane="auto") uses plotty's
    dedicated pane; an explicit target_pane — or PLOTTY_PANE — overrides it with
    the classic resolution (int index, negative-from-the-end, or a named target).
    """
    if target_pane == "auto":
        env_pane = _env("PANE", None)
        if env_pane not in (None, "auto", ""):
            _cfg["auto_pane"] = False
            return _ensure_separate_pane(env_pane, verbose)
        _cfg["auto_pane"] = True              # owned pane: recreate it if killed
        return _ensure_dedicated_pane(verbose)
    _cfg["auto_pane"] = False
    return _ensure_separate_pane(target_pane, verbose)


def _own_pane_is_dedicated():
    """True if the pane this process runs in is a plotty-created dedicated pane
    (tagged `@plotty`). Used by the viewer to decide whether it may close its own
    pane on a user quit — a pane the user gave via target_pane is left alone."""
    pane = os.environ.get("TMUX_PANE")
    if not pane:
        return False
    try:
        out = subprocess.run([_cfg["tmux"], "show", "-pqv", "-t", pane, _DEDICATED_OPT],
                             capture_output=True, text=True, check=False)
    except OSError:
        return False
    return out.stdout.strip() == "1"


def _kill_own_pane():
    """Close the tmux pane this process runs in (best effort)."""
    pane = os.environ.get("TMUX_PANE")
    if not pane:
        return
    try:
        subprocess.run([_cfg["tmux"], "kill-pane", "-t", pane],
                       capture_output=True, check=False)
    except OSError:
        pass


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
    if cmd and cmd != "kitty":
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
    imgval = "kitty" if cmd == "kitty" else "''"  # empty == built-in sixel
    env = (
        f"{_ENV}_IMGCAT={imgval} "
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


def _pane_alive(pane):
    """True if `pane` is still a live tmux pane. Errors are captured, so probing a
    pane the user has since closed is silent (no 'can't find pane' on the REPL)."""
    try:
        out = subprocess.run(
            [_cfg["tmux"], "display-message", "-p", "-t", str(pane), "#{pane_id}"],
            capture_output=True, text=True, check=False)
    except OSError:
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def _refresh_display_pane():
    """Recreate the dedicated plot pane if the user has closed it (auto mode only).

    Without this, every later figure would send-keys / write into a dead pane id
    and tmux would print 'can't find pane: %N'. Returns True when it actually
    recreated the pane — the caller then lets the relaunched viewer draw the
    just-published figure on startup, instead of also signalling/emitting into a
    pane whose shell may still be coming up.
    """
    if not _cfg.get("auto_pane") or os.environ.get("TMUX") is None:
        return False                             # explicit target: user's to manage
    if _pane_alive(_cfg["pane"]):
        return False
    _cfg["made_pane"] = None
    _cfg["pane"] = _ensure_dedicated_pane(_cfg.get("verbose", 0))
    if _cfg.get("can_display", True) and not _cfg["inline"]:
        _ensure_viewer()
    return True


def _write_inline(path):
    """Render sixel without a viewer: to the target tmux pane's tty when in tmux,
    otherwise to this terminal's own stdout."""
    try:
        if os.environ.get("TMUX") is not None:
            tty = _pane_tty(_cfg["pane"])
            if not tty:                          # pane gone: don't dump into the REPL
                _warn_once("inline_tty", "plot pane is gone — re-run enable() to "
                                         "get a new one")
                return
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
    recreated = _refresh_display_pane()          # closed dedicated pane -> new one
    if not _cfg.get("can_display", True):
        return                                   # recreation failed: warn next call
    if _cfg["inline"]:
        _write_inline(_last)
    elif recreated:
        pass                                     # the relaunched viewer draws _last
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
    recreated = _refresh_display_pane()
    if not _cfg.get("can_display", True):
        return
    if _cfg["inline"]:
        if os.path.exists(_last):
            _write_inline(_last)
    elif recreated:
        pass                                     # the relaunched viewer draws _last
    elif not _signal_viewer() and os.path.exists(_last):
        _emit()


# ---- the viewer (runs in the plot pane) -------------------------------------

def view():
    """Viewer loop: redraw on SIGUSR1 (new figure) / SIGWINCH (resize), exit on
    SIGTERM/SIGINT/SIGHUP. When the pane tty is interactive, single keys
    navigate figure history: p/k = older, n/j = newer, q = quit.

    A *user* quit — Ctrl+C (SIGINT) or 'q' — of a plotty-owned dedicated pane
    closes that pane rather than leaving a half-dead shell prompt with a stale
    plot and no navigation; the next figure splits a fresh pane. SIGTERM/SIGHUP
    (disable, viewer restart, or a pane already dying) just exit the process.

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
    sigint = int(getattr(signal, "SIGINT", -1))
    kb_fd = kb_old = None
    offset = 0                                   # 0 = live; k > 0 = k figures back
    kill_pane = False                            # set on a user quit of an owned pane

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
                # A user quit (Ctrl+C, or 'q') of plotty's own pane closes that
                # pane outright, rather than dropping to a shell prompt with a
                # stale plot and no navigation; the next figure splits a fresh
                # one. SIGTERM/SIGHUP (disable / restart / pane already dying)
                # just exit and leave pane management to the backend.
                if ((sigint in sigs) or (b"q" in keys)) and _own_pane_is_dedicated():
                    kill_pane = True
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
    if kill_pane:
        _kill_own_pane()                         # after cleanup: pidfile + termios first
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
        f"{_ENV}_TMUX={shlex.quote(_cfg['tmux'])}",   # so the viewer can run tmux
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


def _tmux_passthrough():
    """Value of tmux's allow-passthrough option ('on'/'all'/'off'), or None."""
    try:
        out = subprocess.run([_cfg["tmux"], "show", "-gv", "allow-passthrough"],
                             capture_output=True, text=True, check=False).stdout
    except OSError:
        return None
    return out.strip() or None


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
        proto = "kitty graphics" if _cfg.get("imgcat") == "kitty" else "sixel"
        print(f"[{name}] inline mode: piping {proto} to {where}", file=sys.stderr)
    if intmux and _cfg.get("imgcat") == "kitty":  # kitty protocol needs passthrough
        ver = _tmux_version()
        if ver is not None and ver < (3, 3):
            print(f"[{name}] tmux {ver[0]}.{ver[1]} is older than 3.3 and has no "
                  f"allow-passthrough; kitty graphics will not display",
                  file=sys.stderr)
        ap = _tmux_passthrough()
        if ap is not None and ap not in ("on", "all"):
            print(f"[{name}] kitty graphics need tmux passthrough; run: tmux set "
                  f"-g allow-passthrough on (and add it to ~/.tmux.conf). Also "
                  f"requires a kitty-protocol terminal (kitty, ghostty); nested "
                  f"tmux is not supported", file=sys.stderr)
    elif intmux:                                 # sixel modes lean on tmux's sixel
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


def _client_termname():
    """TERM of the attached tmux client (e.g. 'xterm-ghostty'), best effort."""
    try:
        out = subprocess.run([_cfg["tmux"], "display-message", "-p",
                              "#{client_termname}"],
                             capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""
    return out.strip()


def _detect_renderer():
    """Pick the built-in encoder for this terminal: sixel when the terminal
    advertises it, else the kitty-graphics encoder; sixel when undeterminable.

    The terminal's *identity* is checked before its advertised features:
    ghostty/kitty never render sixel, while a `terminal-features ',*:sixel'`
    override (the standard nested-tmux setup) makes tmux claim sixel for every
    client — so for those terminals the name is the truthful signal. In tmux
    the *outer* terminal's capabilities are what matter (tmux's own DA1 reply
    advertises whatever tmux was built with); outside tmux, query directly.
    """
    if os.environ.get("TMUX") is not None:
        term = _client_termname()
        if "ghostty" in term or "kitty" in term:
            return "kitty"                       # these never render sixel
        feats = _tmux_features()
        if feats and "sixel" not in feats:
            return "kitty"
        return None                              # sixel (or unknown -> sixel)
    ident = os.environ.get("TERM", "") + " " + os.environ.get("TERM_PROGRAM", "")
    if "ghostty" in ident or "kitty" in ident or os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"
    sixel, kitty = _probe_terminal()
    if not sixel and kitty:
        return "kitty"
    return None                                  # sixel (or unknown -> sixel)


def _resolve_imgcat(imgcat, verbose):
    """Resolve the renderer command, where None means the built-in sixel encoder.

    The default (imgcat=None or "auto", no PLOTTY_IMGCAT) auto-detects the
    terminal's protocol: sixel-capable -> built-in sixel encoder, otherwise the
    built-in kitty-graphics encoder (ghostty/kitty). Explicit overrides:
    "builtin"/""/False -> built-in sixel; "kitty" -> kitty-graphics encoder;
    "chafa"/"img2sixel"/"magick"/"convert" -> that external tool (warning +
    built-in fallback if it isn't installed); any other string -> custom command.
    """
    if imgcat is None:
        imgcat = _env("IMGCAT", None)
    if imgcat in (None, "auto"):                 # default: protocol detection
        imgcat = _detect_renderer()
        if imgcat == "kitty" and verbose:
            print(f"[{__name__}] terminal does not advertise sixel: using the "
                  f"built-in kitty-graphics encoder", file=sys.stderr)
        return imgcat
    if imgcat in ("", "builtin", False):
        return None                              # built-in sixel encoder
    if imgcat == "kitty":
        return "kitty"                           # built-in kitty-graphics encoder
    if _renderer_for(imgcat):                    # bare tool name, e.g. "chafa"
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


def enable(target_pane="auto", imgcat=None, clear=True, tmux="tmux", dpi=None,
           close=True, size=None, bg=None, hist=None, inline=None, viewer=True,
           verbose=1):
    """Activate plotty: detect a renderer, point at a pane, start the viewer.

    target_pane="auto" (default; `PLOTTY_PANE` unset) uses a dedicated plot pane
    in the current tmux window: enable() splits one off the first time, reuses it
    on later calls, and recreates it if you close it — so plotty never draws into
    a pane you opened yourself (an editor, logs, …). Override it with an explicit
    target: an int indexes the window's panes (negative counts from the end,
    `-1` = last), or pass a tmux target name like "sess:win.pane". `PLOTTY_PANE`
    sets the default and, when set, also overrides "auto".

    inline=None (default) auto-selects: inline mode when not in tmux, viewer-pane
    mode when in tmux. In inline mode the backend renders sixel itself (no viewer
    process) and writes it to the target pane's tty when in tmux, or to this
    terminal's stdout when not. inline=True forces inline even inside tmux;
    inline=False forces viewer-pane mode. `PLOTTY_INLINE=1/0` sets the default.

    imgcat=None (default, same as "auto") detects the terminal's protocol:
    sixel-capable terminals get the built-in sixel encoder, others (ghostty,
    kitty) get the built-in kitty-graphics encoder (Unicode placeholders —
    robust inside a single tmux; needs `tmux set -g allow-passthrough on`; not
    nested tmux). Overrides: imgcat="builtin" forces the sixel encoder,
    imgcat="kitty" forces the kitty encoder, imgcat="chafa" / "img2sixel" /
    "magick" uses that external sixel tool (slightly faster, better resampling;
    falls back to built-in with a warning if it isn't installed), and any other
    string is used verbatim as a custom command. `PLOTTY_IMGCAT` sets the
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
    _cfg["verbose"] = verbose                    # reused when recreating a killed pane

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
            _cfg["pane"] = _resolve_display_pane(target_pane, verbose)
        elif inline is not True and _cfg["imgcat"] != "kitty":
            # auto-selected inline: verify the terminal renders sixel (the kitty
            # encoder is an explicit opt-in and doesn't need the sixel attribute)
            if _stdout_supports_sixel() is False:
                _cfg["can_display"] = False
                if verbose:
                    print(f"[{__name__}] this terminal does not appear to support "
                          f"sixel — figures will not be displayed (use a "
                          f"sixel-capable terminal or tmux; enable(inline=True) "
                          f"forces output anyway)", file=sys.stderr)
    else:
        _cfg["pane"] = _resolve_display_pane(target_pane, verbose)
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
    renderer = _cfg["imgcat"] or "built-in sixel encoder"
    if renderer == "kitty":
        renderer = "built-in kitty graphics (unicode placeholders)"
    lines = [
        f"plotty {__version__}",
        f"  mode:      {mode}",
        f"  renderer:  {renderer}",
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
