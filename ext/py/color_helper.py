import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd

import sklearn.neighbors
from scipy.sparse import coo_array
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
matplotlib.use('Agg')

def assign_color_tsp(
    mtx: np.ndarray,
    cmap_name: str = "nipy_spectral",
    weight: np.ndarray = None,
    two_opt: bool = True,
    spectral_offset: float = 0.05,
    anchor_color: str = None,
    anchor_grid: int = 1024
) -> np.ndarray:
    K = mtx.shape[0]
    assert mtx.shape == (K, K)

    # --- normalize weight or use uniform ---
    if weight is None:
        weight = np.ones(K) / K
    else:
        weight = weight.astype(float)
        weight /= weight.sum()

    # --- build a "distance" matrix D = normalized similarity ---
    D = mtx.astype(float).copy()
    D /= D.max()

    # --- greedy TSP nearest-neighbour (minimize D) ---
    start = int(weight.argmax())
    cycle = [start]
    unvisited = set(range(K)) - {start}
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[current, j])
        cycle.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    # --- 2-opt improvement ---
    if two_opt:
        improved = True
        while improved:
            improved = False
            for i in range(1, K - 2):
                for j in range(i + 1, K):
                    a, b = cycle[i - 1], cycle[i]
                    c, d = cycle[j - 1], cycle[j % K]
                    if D[a, b] + D[c, d] > D[a, c] + D[b, d]:
                        cycle[i:j] = reversed(cycle[i:j])
                        improved = True
    print(cycle)
    # --- even‐spaced angular positions on [0,1) ---
    c_pos = np.empty(K, float)
    for rank, idx in enumerate(cycle):
        c_pos[idx] = rank / K

    cmap = plt.get_cmap(cmap_name)
    # --- if the user wants to pin one factor to an anchor color ---
    if anchor_color is not None:
        # pick which factor to pin
        anchor_index = int(weight.argmax())
        # convert hex to RGB
        target_rgb = np.array(mcolors.to_rgb(anchor_color))[None, :]
        # sample the colormap finely
        grid = np.linspace(0, 1, anchor_grid)
        grid_rgb = cmap(grid)[:, :3]
        # compute Euclidean distance in RGB‐space
        dists = np.linalg.norm(grid_rgb - target_rgb, axis=1)
        # find the grid position whose color best matches the anchor
        t_anchor = grid[dists.argmin()]
        # shift all c_pos so that anchor_index lands on t_anchor
        delta = t_anchor - c_pos[anchor_index]
        c_pos = (c_pos + delta) % 1.0

    # --- inset by spectral_offset and renormalize ---
    c_pos = c_pos * (1 - 2*spectral_offset) + spectral_offset

    # --- finally sample the colormap to get RGB colors ---
    rgb = cmap(c_pos)[:, :3]

    return rgb


def plot_colortable(colors, title, sort_colors=True, ncols=4, dpi = 80,\
                    cell_width = 212, cell_height = 22,\
                    title_fontsize=24, text_fontsize=24,\
                    swatch_width = 48, margin = 12, topmargin = 40):
    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + margin + topmargin


    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=title_fontsize, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=text_fontsize,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

def choose_color(_args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--cmap-name', type=str, default="nipy_spectral", help="Name of Matplotlib colormap to use (better close to a circular colormap)")
    parser.add_argument('--top-color', type=str, default="#fcd217", help="HEX color code for the top factor")
    parser.add_argument('--even-space', action='store_true', help="Evenly space the factors on the spectrum")
    parser.add_argument("--skip-columns", type=str, action="append", default=[], help="Columns that are neither coordiante nor factor in the input file")
    parser.add_argument('--annotation', type=str, default = '', help='')
    parser.add_argument('--seed', type=int, default=-1, help='')
    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return

    ## obtain seed if not provided
    seed = args.seed
    if seed <= 0:
        seed = int(datetime.now().timestamp()) % 2147483648
    np.random.seed(seed)

    factor_name = {}
    if os.path.isfile(args.annotation):
        with open(args.annotation) as f:
            for line in f:
                x = line.strip().split('\t')
                factor_name[x[0]] = x[1]

    cmap_name = args.cmap_name
    if args.cmap_name not in plt.colormaps():
        cmap_name = "nipy_spectral"

    df = pd.read_csv(args.input, sep='\t', header=0)
    df.rename(columns = {"X":"x","Y":"y"},inplace=True)
    header = df.columns
    factor_header = [x for x in header if x not in ["x", "y"] + args.skip_columns]
    K = len(factor_header)
    N = df.shape[0]

    # Factor abundance (want top factors to have more distinct colors)
    if args.even_space:
        weight=None
    else:
        weight = df.loc[:, factor_header].sum(axis = 0).values
        weight = weight**(1/2)
        weight /= weight.sum()
        weight = np.clip(weight, .5/K, 4/K)
        weight /= weight.sum()

    # Find neearest neighbors
    bt = sklearn.neighbors.BallTree(df.loc[:, ["x", "y"]])
    dist, indx = bt.query(df.loc[:, ["x", "y"]], k = 7, return_distance=True)
    r_indx = np.array([i for i,v in enumerate(indx) for y in range(len(v))], dtype=int)
    c_indx = indx.reshape(-1)
    dist = dist.reshape(-1)
    nn = dist[dist > 0].min()
    mask = (dist < nn + .5)
    r_indx = r_indx[mask]
    c_indx = c_indx[mask]
    # Compute spatial similarity
    Sig = coo_array((np.ones(len(r_indx)), (r_indx, c_indx)), shape=(N, N)).tocsr()
    W = np.array(df.loc[:, factor_header])
    mtx = W.T @ Sig @ W + 1e-6
    # set diagonal to 0
    np.fill_diagonal(mtx, 0)
    # row normalize
    mtx = mtx / np.sqrt(np.sum(mtx, axis=1, keepdims=True))
    # take the element-wise max of mtx and mtx.T
    mtx = np.maximum(mtx, mtx.T)
    # Large values in mtx indicate close proximity, to be mapped to distinct colors

    cmtx = assign_color_tsp(mtx, cmap_name=cmap_name, weight=weight, two_opt=True, spectral_offset=0.05, anchor_color=args.top_color)
    # translate RGB to 0-255
    cmtx_int = (cmtx * 255).astype(int) # K x 3
    # translate each RGB color to hex code
    chex = ['#%02x%02x%02x' % (r, g, b) for r, g, b in cmtx_int]
    df = pd.DataFrame({"Color_hex":chex, "Name":factor_header})
    df = pd.concat([pd.DataFrame(cmtx_int, columns=["R", "G", "B"]), df], axis=1)
    cdict = {v:cmtx[k,:] for k,v in enumerate(factor_header)}
    if len(factor_name) > 0:
        df["Annotation"] = df.Name.map(factor_name)
        cdict = {factor_name[v]:cmtx[k,:] for k,v in enumerate(factor_header) }

    # Output RGB table
    f = args.output + ".rgb.tsv"
    df.to_csv(f, sep='\t', index=False)

    # Plot color bar
    fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
    f = args.output + ".cbar.png"
    fig.savefig(f, format="png", transparent=True)

if __name__ == "__main__":
    choose_color(sys.argv[1:])
