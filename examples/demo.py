#!/usr/bin/env python3
"""plotty demo — generate a series of matplotlib plots and display them.

Run it inside a tmux session (split off a plot pane first, e.g. Ctrl-b "):

    python examples/demo.py

Each figure is drawn in the plot pane (or inline, if you're not in tmux);
press Enter to advance to the next one. Uses only numpy + matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

import plotty


def pause(msg="next"):
    try:
        input(f"  [Enter] {msg} ... ")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit


def main():
    plotty.enable()                      # auto-detect renderer, target the last pane
    rng = np.random.default_rng(0)

    # 1. line plot
    x = np.linspace(0, 4 * np.pi, 400)
    plt.figure()
    plt.plot(x, np.sin(x), label="sin")
    plt.plot(x, np.cos(x), label="cos")
    plt.title("line plot")
    plt.legend()
    plt.show()
    pause()

    # 2. scatter with a colour map
    plt.figure()
    plt.scatter(rng.normal(size=300), rng.normal(size=300),
                c=rng.random(300), s=20, cmap="viridis")
    plt.title("scatter")
    plt.colorbar()
    plt.show()
    pause()

    # 3. histogram
    plt.figure()
    plt.hist(rng.normal(size=5000), bins=40, color="steelblue")
    plt.title("histogram")
    plt.show()
    pause()

    # 4. 2x2 panel of mixed plot types
    fig, ax = plt.subplots(2, 2, figsize=(7, 5))
    ax[0, 0].plot(x, np.sin(x)); ax[0, 0].set_title("sin")
    ax[0, 1].plot(x, np.exp(-x / 5) * np.sin(x)); ax[0, 1].set_title("damped")
    ax[1, 0].bar(range(5), rng.random(5)); ax[1, 0].set_title("bar")
    ax[1, 1].imshow(rng.random((20, 20)), cmap="magma"); ax[1, 1].set_title("imshow")
    fig.suptitle("subplots")
    fig.tight_layout()
    plt.show()
    pause("finish")

    plotty.disable()
    print("done.")


if __name__ == "__main__":
    main()
