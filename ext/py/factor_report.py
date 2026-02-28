import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader


COL_FACTOR = "Factor"
COL_FEATURE = "Feature"
COL_FC = "FoldChange"
COL_PVAL = "pval"
COL_LOG10PVAL = "log10pval"


def _norm_factor_id(v):
    s = str(v).strip()
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def _find_column(columns, candidates):
    if not candidates:
        return None
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    lower_map = {str(c).lower(): c for c in columns}
    for c in candidates:
        key = str(c).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _read_table(path, what):
    if not path or not os.path.exists(path):
        sys.exit(f"Cannot find {what}: {path}")
    try:
        return pd.read_csv(path, sep="\t")
    except Exception as ex:
        sys.exit(f"Failed to read {what}: {path}\n{ex}")


def _normalize_color_table(path):
    if not path or not os.path.isfile(path):
        sys.exit(f"Cannot find color table: {path}")

    # Flexible parsing for tab/space/comma separated color files.
    try:
        df = pd.read_csv(path, sep=r"[\t,\s]+", engine="python", comment="#")
    except Exception as ex:
        sys.exit(f"Failed to read color table: {path}\n{ex}")

    rcol = _find_column(df.columns, ["R", "Red"])
    gcol = _find_column(df.columns, ["G", "Green"])
    bcol = _find_column(df.columns, ["B", "Blue"])

    if rcol is None or gcol is None or bcol is None:
        # Retry assuming no header.
        try:
            df = pd.read_csv(path, sep=r"[\t,\s]+", engine="python", comment="#", header=None)
        except Exception as ex:
            sys.exit(f"Failed to parse color table as headerless file: {path}\n{ex}")
        if df.shape[1] < 3:
            sys.exit("Color table must contain at least 3 columns (R G B)")
        rgb = df.iloc[:, :3].copy()
        rgb.columns = ["R", "G", "B"]
    else:
        rgb = df.loc[:, [rcol, gcol, bcol]].copy()
        rgb.columns = ["R", "G", "B"]

    for c in ["R", "G", "B"]:
        rgb[c] = pd.to_numeric(rgb[c], errors="coerce")
    rgb.dropna(inplace=True)
    if rgb.empty:
        sys.exit("No valid RGB rows found in color table")

    rgb = rgb.astype(int).clip(0, 255).reset_index(drop=True)
    rgb["RGB"] = rgb[["R", "G", "B"]].astype(str).agg(",".join, axis=1)
    rgb["HEX"] = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb[["R", "G", "B"]].values]
    return rgb


def _normalize_pseudobulk(path, feature_label):
    post = _read_table(path, "pseudobulk file")

    feature_col = _find_column(post.columns, [feature_label, "Feature", "feature", "Gene", "gene"])
    if feature_col is None:
        feature_col = post.columns[0]
        logging.warning(
            "Cannot find feature column by name; using first column '%s' as feature label",
            feature_col,
        )

    factor_cols = [c for c in post.columns if c != feature_col]
    if not factor_cols:
        sys.exit("Pseudobulk file has no factor columns")

    kept = []
    for c in factor_cols:
        v = pd.to_numeric(post[c], errors="coerce")
        if v.notna().any():
            post[c] = v.fillna(0.0)
            kept.append(c)
        else:
            logging.warning("Dropping non-numeric pseudobulk column: %s", c)

    if not kept:
        sys.exit("No numeric factor columns found in pseudobulk file")

    post[feature_col] = post[feature_col].astype(str)
    return post, feature_col, kept


def _normalize_de(path, feature_label, factor_label):
    df = _read_table(path, "DE file")

    factor_col = _find_column(df.columns, [factor_label, "Factor", "factor", "Topic", "topic"])
    if factor_col is None:
        sys.exit(f"Cannot find factor column in DE file: {path}")

    feature_col = _find_column(df.columns, [feature_label, "Feature", "feature", "Gene", "gene"])
    if feature_col is None:
        logging.warning("Cannot find feature column in DE file; using row index as feature labels")
        feat = df.index.astype(str)
    else:
        feat = df[feature_col].astype(str).fillna("")

    fc_col = _find_column(
        df.columns,
        ["FoldChange", "ApproxFC", "approxFC", "FC", "fc", "log2FC", "Log2FC"],
    )
    if fc_col is None:
        logging.warning("Cannot find fold-change column in DE file; filling FoldChange with 0")
        fc = pd.Series(0.0, index=df.index)
    else:
        fc = pd.to_numeric(df[fc_col], errors="coerce").fillna(0.0)

    log10_col = _find_column(
        df.columns,
        ["log10pval", "logPval", "LogPval", "-log10pval", "neglog10pval", "neglog10p"],
    )
    pval_col = _find_column(df.columns, ["pval", "Pval", "pValue", "PValue", "p_value", "p.value"])
    chi2_col = _find_column(df.columns, ["Chi2", "chi2", "ChiSq", "chisq", "chi_sq"])

    if log10_col is not None:
        log10p = pd.to_numeric(df[log10_col], errors="coerce")
    elif pval_col is not None:
        pval = pd.to_numeric(df[pval_col], errors="coerce").fillna(1.0)
        log10p = -np.log10(np.clip(pval.values, 1e-300, 1.0))
        log10p = pd.Series(log10p, index=df.index)
    else:
        logging.warning("Cannot find p-value/log10p column in DE file; filling log10pval with NaN")
        log10p = pd.Series(np.nan, index=df.index)

    if chi2_col is not None:
        chi2 = pd.to_numeric(df[chi2_col], errors="coerce")
    else:
        chi2 = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(
        {
            COL_FACTOR: df[factor_col].map(_norm_factor_id),
            COL_FEATURE: feat,
            COL_FC: fc,
            COL_LOG10PVAL: log10p,
            "Chi2": chi2,
        }
    )

    if out[COL_LOG10PVAL].notna().any():
        sort_col = COL_LOG10PVAL
    elif out["Chi2"].notna().any():
        sort_col = "Chi2"
    else:
        sort_col = COL_FC
        logging.warning("No log10pval/Chi2 found in DE file; ranking by FoldChange")

    return out, sort_col


def _top_genes_by_metric(df, factor_names, sort_col, ntop, mtop, min_log10p, min_fc):
    work = df.copy()
    work.sort_values(by=[COL_FACTOR, sort_col], ascending=False, inplace=True, na_position="last")
    work["Rank"] = (
        work.groupby(COL_FACTOR)[sort_col]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )

    keep = (work["Rank"] < mtop) | (
        work[COL_LOG10PVAL].gt(min_log10p) & work[COL_FC].ge(min_fc)
    )

    out = []
    for kname in factor_names:
        sub = work.loc[work[COL_FACTOR].eq(str(kname)) & keep, COL_FEATURE].iloc[:ntop]
        vals = sub.astype(str).tolist()
        out.append(", ".join(vals) if vals else ".")
    return out


def factor_report(_args):
    parser = argparse.ArgumentParser(prog="factor_report")
    parser.add_argument("--de", type=str, required=True, help="")
    parser.add_argument("--de_neighbor", type=str, default="", help="")
    parser.add_argument("--pseudobulk", type=str, required=True, help="")
    parser.add_argument("--feature_label", type=str, default="Feature", help="")
    parser.add_argument("--factor_label", type=str, default="Factor", help="")
    parser.add_argument("--color_table", type=str, required=True, help="")
    parser.add_argument("--n_top_gene", type=int, default=20, help="")
    parser.add_argument("--min_top_gene", type=int, default=10, help="")
    parser.add_argument("--max_pval", type=float, default=0.001, help="")
    parser.add_argument("--min_fc", type=float, default=1.5, help="")
    parser.add_argument("--output_pref", type=str, required=True, help="")
    parser.add_argument("--annotation", type=str, default="", help="")
    parser.add_argument("--anchor", type=str, default="", help="")
    parser.add_argument("--keep_order", action="store_true", help="")
    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ntop = max(1, args.n_top_gene)
    mtop = max(1, args.min_top_gene)
    min_log10p = -np.log10(max(args.max_pval, 1e-300))
    min_fc = args.min_fc

    ejs = os.path.join(os.path.dirname(__file__), "factor_report.template.html")
    if not os.path.isfile(ejs):
        sys.exit(f"Template file {ejs} not found")

    color_table = _normalize_color_table(args.color_table)
    post, post_feature_col, factor_header = _normalize_pseudobulk(args.pseudobulk, args.feature_label)
    de, de_sort_col = _normalize_de(args.de, args.feature_label, args.factor_label)

    neighbor_de = None
    neighbor_sort_col = None
    if args.de_neighbor:
        neighbor_de, neighbor_sort_col = _normalize_de(
            args.de_neighbor, args.feature_label, args.factor_label
        )

    k = len(factor_header)
    if len(color_table) < k:
        logging.warning(
            "Color table has %d colors, less than %d factors; cycling colors",
            len(color_table),
            k,
        )
        rep = (k // len(color_table)) + 1
        color_table = pd.concat([color_table] * rep, ignore_index=True).iloc[:k, :].copy()
    else:
        color_table = color_table.iloc[:k, :].copy()
    color_table.reset_index(drop=True, inplace=True)

    factor_cols = list(factor_header)
    factor_header = [_norm_factor_id(x) for x in factor_cols]
    post_umi = post.loc[:, factor_cols].sum(axis=0).astype(float).values.astype(int)
    post_weight = post.loc[:, factor_cols].sum(axis=0).astype(float).values
    total_weight = float(post_weight.sum())
    if total_weight > 0:
        post_weight = post_weight / total_weight
    else:
        post_weight = np.zeros_like(post_weight)
        logging.warning("Total pseudobulk weight is zero; output weights set to zero")

    top_gene_pval = _top_genes_by_metric(
        de, factor_header, de_sort_col, ntop, mtop, min_log10p, min_fc
    )

    top_gene_specific = None
    if neighbor_de is not None:
        top_gene_specific = _top_genes_by_metric(
            neighbor_de,
            factor_header,
            neighbor_sort_col,
            ntop,
            mtop,
            min_log10p,
            min_fc,
        )

    top_gene_fc = _top_genes_by_metric(
        de, factor_header, COL_FC, ntop, mtop, min_log10p, min_fc
    )

    top_gene_weight = []
    feature_values = post[post_feature_col].astype(str).values
    for i, (kname, kcol) in enumerate(zip(factor_header, factor_cols)):
        if post_umi[i] < 10:
            top_gene_weight.append(".")
            continue
        order = np.argsort(-post.loc[:, kcol].values)[:ntop]
        vals = [feature_values[j] for j in order]
        top_gene_weight.append(", ".join(vals) if vals else ".")

    table = pd.DataFrame(
        {
            "Factor": factor_header,
            "RGB": color_table["RGB"].values,
            "Weight": post_weight,
            "PostUMI": post_umi,
            "TopGene_pval": top_gene_pval,
            "TopGene_fc": top_gene_fc,
            "TopGene_weight": top_gene_weight,
        }
    )
    if top_gene_specific is not None:
        table.insert(table.columns.get_loc("TopGene_pval") + 1, "TopGene_specific", top_gene_specific)

    out_header = ["Factor", "RGB", "Weight", "PostUMI", "TopGene_pval"]
    if top_gene_specific is not None:
        out_header.append("TopGene_specific")
    out_header += ["TopGene_fc", "TopGene_weight"]

    if args.anchor and os.path.exists(args.anchor):
        ak = pd.read_csv(args.anchor, sep="\t", comment="#", header=None)
        if ak.shape[1] >= 2:
            ak = ak.iloc[:, :2].copy()
            ak.columns = ["Factor", "Anchors"]
            ak["Factor"] = ak["Factor"].map(_norm_factor_id)
            table = table.merge(ak, on="Factor", how="left")
            out_header.insert(4, "Anchors")
            logging.info("Read anchor genes from %s", args.anchor)
        else:
            logging.warning("Anchor file has fewer than 2 columns; ignoring: %s", args.anchor)

    if not args.keep_order:
        table.sort_values(by="Weight", ascending=False, inplace=True)

    if args.annotation and os.path.isfile(args.annotation):
        anno_df = pd.read_csv(args.annotation, sep="\t", comment="#", header=None)
        if anno_df.shape[1] >= 2:
            anno_map = {
                _norm_factor_id(k): str(v)
                for k, v in zip(anno_df.iloc[:, 0].values, anno_df.iloc[:, 1].values)
            }
            table["Factor"] = table["Factor"].map(lambda x: anno_map.get(_norm_factor_id(x), str(x)))
        else:
            logging.warning(
                "Annotation file has fewer than 2 columns; ignoring: %s", args.annotation
            )

    out_tsv = args.output_pref + ".tsv"
    table.loc[table["PostUMI"].ge(10), out_header].to_csv(
        out_tsv, sep="\t", index=False, header=True, float_format="%.5f"
    )

    with open(out_tsv, "r") as rf:
        lines = rf.readlines()
    header = lines[0].strip().split("\t")
    rows = [list(enumerate(row.strip().split("\t"))) for row in lines[1:]]

    env = Environment(loader=FileSystemLoader(os.path.dirname(ejs)))
    template = env.get_template(os.path.basename(ejs))
    html_output = template.render(
        header=header,
        rows=rows,
        image_base64=None,
        tree_image_alt=None,
        tree_image_caption=None,
    )

    out_html = args.output_pref + ".info.html"
    with open(out_html, "w") as html_file:
        html_file.write(html_output)

    print(out_html)


if __name__ == "__main__":
    factor_report(sys.argv[1:])
