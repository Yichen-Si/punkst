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

def _rgb_to_lab(rgb):
    """rgb: (M,3) in [0,1] → Lab: (M,3)"""
    # 1) linearize
    def to_lin(c):
        return np.where(c <= 0.04045,
                        c / 12.92,
                        ((c + 0.055) / 1.055) ** 2.4)
    rgb_lin = to_lin(rgb)

    # 2) lin RGB → XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    XYZ = rgb_lin @ M.T

    # 3) normalize by reference white
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = XYZ / white

    # 4) f(t) for Lab
    delta = 6/29
    def f(t):
        return np.where(t > delta**3,
                        np.cbrt(t),
                        t/(3*delta**2) + 4/29)
    f_xyz = f(xyz)

    L = 116*f_xyz[:,1] - 16
    a = 500*(f_xyz[:,0] - f_xyz[:,1])
    b = 200*(f_xyz[:,1] - f_xyz[:,2])
    return np.stack([L, a, b], axis=1)

def _tsp_cycle(D, start=None, two_opt=True, seed=None):
    N = D.shape[0]
    if start is None:
        rng = np.random.RandomState(seed)
        start = int(rng.randint(N))
    cycle = [start]
    unvisited = set(range(N)) - {start}
    curr = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[curr, j])
        cycle.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    if two_opt:
        improved = True
        while improved:
            improved = False
            for i in range(1, N - 2):
                for j in range(i + 1, N):
                    a, b = cycle[i-1], cycle[i]
                    c, d = cycle[j-1], cycle[j % N]
                    if D[a,b] + D[c,d] > D[a,c] + D[b,d]:
                        cycle[i:j] = reversed(cycle[i:j])
                        improved = True
    return cycle

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
    cycle = _tsp_cycle(D, start=start, two_opt=two_opt)
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

def assign_color_from_table(
    mtx: np.ndarray,
    color_df: pd.DataFrame,
    weight: np.ndarray = None,
    two_opt: bool = True
) -> np.ndarray:
    """
    Pick K colors from a pool of M ≥ K in color_df, so that
    items with high similarity in `mtx` get maximally distinct colors,
    using Lab-space ΔE₇₆ for color-color distance.
    Returns rgb[K,3] floats in [0,1].
    """
    K = mtx.shape[0]

    # --- parse candidate colors ---
    if {"R","G","B"}.issubset(color_df.columns):
        rgb_cand = color_df[["R","G","B"]].values.astype(float)
        if rgb_cand.max() > 1.0:
            rgb_cand /= 255.0
    elif "Color_hex" in color_df.columns:
        rgb_cand = np.vstack(color_df["Color_hex"].map(mcolors.to_rgb).values)
    else:
        raise ValueError("color table needs R,G,B or Color_hex")

    M = rgb_cand.shape[0]
    if M < K:
        raise ValueError(f"Need at least {K} colors, got {M}")
    if M > K:
        rgb_cand = rgb_cand[:K]

    # --- compute Lab for candidates and ΔE₇₆ distance matrix ---
    lab_cand = _rgb_to_lab(rgb_cand)
    Dc = np.linalg.norm(lab_cand[:,None,:] - lab_cand[None,:,:], axis=2)
    Dc /= Dc.max()

    # --- cycle the colors and the items ---
    color_cycle = _tsp_cycle(Dc, start = 0, two_opt=two_opt)

    Di = mtx.astype(float).copy()
    Di /= Di.max()
    start_i = int(weight.argmax()) if (weight is not None) else None
    item_cycle = _tsp_cycle(Di, start=start_i, two_opt=two_opt)

    # --- assign evenly along the color‐cycle ---
    rgb = np.empty((K,3), float)
    for pos, item_idx in enumerate(item_cycle):
        cidx = color_cycle[pos]
        rgb[item_idx] = rgb_cand[cidx]

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
    parser.add_argument('--cmap-name', type=str, help="Name of Matplotlib colormap to use (better close to a circular colormap)")
    parser.add_argument('--color-table', type=str, help="TSV with R,G,B or Color_hex of candidate colors")
    parser.add_argument('--top-color', type=str, default="#fcd217", help="HEX color code for the top factor")
    parser.add_argument('--even-space', action='store_true', help="Evenly space the factors on the spectrum")
    parser.add_argument("--skip-columns", type=str, action="append", default=["random_key", "layer"], help="Columns that are neither coordiante nor factor in the input file")
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

    df = pd.read_csv(args.input, sep='\t', header=0)
    df.rename(columns = {"X":"x","Y":"y"},inplace=True)
    header = df.columns
    factor_header = [x for x in header if x not in ["x", "y"] + args.skip_columns]
    K = len(factor_header)
    N = df.shape[0]

    if K == 0:
        print("Input file does not contain any factors")
        return
    if N < 2:
        print("Input file does not contain enough data")
        return

    # Factor abundance (want top factors to have more distinct colors)
    if args.even_space:
        weight=None
    else:
        weight = df.loc[:, factor_header].sum(axis = 0).values
        weight = weight**(1/2)
        weight /= weight.sum()
        weight = np.clip(weight, .5/K, 4/K)
        weight /= weight.sum()

    # Create a similarity matrix among factors

    if "x" not in header or "y" not in header:
        print("Input file does not seem to contain spatial units (no spatial coordinates are found)")
        # compute cosine similarity among the columns
        from sklearn.metrics.pairwise import cosine_similarity
        mtx = cosine_similarity(df.loc[:, factor_header].values.T)
    else:
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

    if not args.color_table and K <= 48:
        # use default colortable
        # get path to the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.color_table = os.path.join(script_dir, "cmap.48.tsv")
        print("Using default color table", args.color_table)

    if args.color_table:
        ct = pd.read_csv(args.color_table, sep='\t', header=0)
        cmtx = assign_color_from_table (
            mtx, ct,
            weight=weight,
            two_opt=True
        )
    else:
        cmap_name = args.cmap_name
        if args.cmap_name not in plt.colormaps():
            cmap_name = "nipy_spectral"
        print("Using colormap", cmap_name)
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
