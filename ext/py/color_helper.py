import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd

import sklearn.neighbors
from scipy.sparse import coo_array
from datetime import datetime
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

def assign_color_mds_line(mtx, cmap_name, weight=None, top_color=None, seed=None):
    # mtx is a K by K similarity/proximity matrix
    assert mtx.shape[0] == mtx.shape[1], "mtx must be square"
    K = mtx.shape[0]
    # weight is a K vector of factor abundance
    if weight is None:
        weight = np.ones(K)
    weight /= weight.sum()
    # The color of the top factor (the one with the largest weight)
    if top_color is None:
        top_color = "#fcd217"
    else:
        match = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', top_color)
        if match is None:
            top_color = "#fcd217"
    # Find the offset to map the top factor to the desired color
    cgrid = 200
    cmtx=plt.get_cmap(cmap_name)(np.arange(cgrid)/cgrid)
    h = top_color.lstrip('#')
    top_color_rgb = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    d = np.abs(cmtx[:, :3] - np.array(top_color_rgb).reshape((1, -1)) ).sum(axis = 1)
    anchor_pos = d.argmin() / cgrid
    anchor_angle = anchor_pos * 2 * np.pi

    mds = MDS(n_components=1, dissimilarity="precomputed", random_state=seed)
    mds_coordinates = mds.fit_transform(mtx).flatten()
    c_order = np.argsort(np.argsort(mds_coordinates))
    w_vec = weight[np.argsort(mds_coordinates)]
    w_vec = np.cumsum(w_vec) - w_vec/2
    normalized_coordinates = np.zeros(K)
    normalized_coordinates[c_order] = w_vec
    angle = normalized_coordinates * 2 * np.pi
    anchor_k = np.argmax(weight)
    angle_shift = angle + (anchor_angle - angle[anchor_k])
    if angle_shift.max() > 2*np.pi:
        angle_shift -= np.pi * 2
    angle_shift[angle_shift < 0] = 2 * np.pi + angle_shift[angle_shift < 0]
    c_pos = angle_shift / np.pi / 2
    return c_pos

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
    parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use (better close to a circular colormap)")
    parser.add_argument('--top_color', type=str, default="#fcd217", help="HEX color code for the top factor")
    parser.add_argument('--even_space', action='store_true', help="Evenly space the factors on the circle")
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
        cmap_name = "turbo"

    df = pd.read_csv(args.input, sep='\t', header=0)
    df.rename(columns = {"X":"x","Y":"y"},inplace=True)
    header = df.columns
    factor_header = [x for x in header if x not in ["x", "y"]]
    K = len(factor_header)
    N = df.shape[0]

    # Factor abundance (want top factors to have more distinct colors)
    if args.even_space:
        weight=None
    else:
        weight = df.loc[:, factor_header].sum(axis = 0).values
        weight = weight**(1/2)
        weight /= weight.sum()
        weight = np.clip(weight, .2/K, 1)
        weight /= weight.sum()

    # Find neearest neighbors
    bt = sklearn.neighbors.BallTree(df.loc[:, ["x", "y"]])
    dist, indx = bt.query(df.loc[:, ["x", "y"]], k = 7, return_distance=True)
    r_indx = np.array([i for i,v in enumerate(indx) for y in range(len(v))], dtype=int)
    c_indx = indx.reshape(-1)
    dist = dist.reshape(-1)
    nn = dist[dist > 0].min()
    mask = (dist < nn + .5) & (dist > 0)
    r_indx = r_indx[mask]
    c_indx = c_indx[mask]
    # Compute spatial similarity
    Sig = coo_array((np.ones(len(r_indx)), (r_indx, c_indx)), shape=(N, N)).tocsr()
    W = np.array(df.loc[:, factor_header])
    mtx = W.T @ Sig @ W
    # Translate into a symmetric similarity measure
    # Large values in mtx indicate close proximity, to be mapped to distinct colors
    np.fill_diagonal(mtx, 0)
    mtx /= mtx.sum(axis = 1)
    mtx = mtx + mtx.T

    c_pos = assign_color_mds_line(mtx, cmap_name, weight=weight, top_color=args.top_color, seed=seed)

    spectral_offset = .05 # avoid extremely dark colors
    c_pos = (c_pos - c_pos.min()) / (c_pos.max() - c_pos.min()) * (1 - spectral_offset) + spectral_offset

    c_rank = np.argsort(np.argsort(c_pos))
    cmtx = plt.get_cmap(cmap_name)(c_pos)[:, :3] # K x 3
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
