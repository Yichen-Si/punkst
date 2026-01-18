import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.colors
from jinja2 import Environment, FileSystemLoader

def factor_report(_args):

    parser = argparse.ArgumentParser(prog="factor_report")
    parser.add_argument('--de', type=str, help='')
    parser.add_argument('--de_neighbor', type=str, default='', help='')
    parser.add_argument('--pseudobulk', type=str, help='')
    parser.add_argument('--feature_label', type=str, default="Feature", help='')
    parser.add_argument('--color_table', type=str, default='', help='')
    parser.add_argument('--n_top_gene', type=int, default=20, help='')
    parser.add_argument('--min_top_gene', type=int, default=10, help='')
    parser.add_argument('--max_pval', type=float, default=0.001, help='')
    parser.add_argument('--min_fc', type=float, default=1.5, help='')
    parser.add_argument('--output_pref', type=str, help='')
    parser.add_argument('--annotation', type=str, default = '', help='')
    parser.add_argument('--anchor', type=str, default='', help='')
    parser.add_argument('--keep_order', action='store_true', help='')
    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return
    ntop = args.n_top_gene
    mtop = args.min_top_gene

    # Template
    ejs = os.path.join(os.path.dirname(__file__), "factor_report.template.html")
    if not os.path.isfile(ejs):
        sys.exit(f"Template file {ejs} not found")
    # Color code
    if not os.path.isfile(args.color_table):
        sys.exit(f"Cannot find color table")
    color_table = pd.read_csv(args.color_table, sep='\t')
    # Posterior count
    if not os.path.exists(args.pseudobulk):
        sys.exit(f"Cannot find posterior count file")
    post = pd.read_csv(args.pseudobulk, sep='\t')
    def load_de(path):
        if not os.path.exists(path):
            sys.exit(f"Cannot find DE file")
        df = pd.read_csv(path, sep='\t', dtype={'Factor':str})
        df.rename(columns = {"logPval":"log10pval", "ApproxFC":"FoldChange", "gene":"Feature", "Gene":"Feautre", "Pval":"pval"}, inplace=True)
        sortby = "log10pval"
        if sortby not in df.columns:
            sortby = "Chi2"
        if "log10pval" not in df.columns:
            df["log10pval"] = -np.log10(np.clip(df["pval"].values, 1e-300, 1.0))
        return df, sortby

    # DE genes
    de, sortby = load_de(args.de)
    neighbor_de = None
    neighbor_sortby = None
    if args.de_neighbor:
        neighbor_de, neighbor_sortby = load_de(args.de_neighbor)

    output_pref = args.output_pref
    min_log10p = -np.log10(args.max_pval)

    factor_header = list(post.columns[1:])
    for u in factor_header:
        post[u] = post[u].astype(float)
    K = len(factor_header)

    color_table = color_table.iloc[:len(factor_header), :]
    color_table['RGB'] = [','.join(x) for x in np.clip((color_table.loc[:, ['R','G','B']].values).astype(int), 0, 255).astype(str) ]
    color_table['HEX'] = [ matplotlib.colors.to_hex(v) for v in np.clip(color_table.loc[:, ['R','G','B']].values / 255, 0, 1) ]
    if len(color_table) < K:
        logging.warning(f"Color table has only {len(color_table)} colors, less than {K} factors")
        # cycle rows in color table
        color_table = pd.concat( [color_table]*((K // len(color_table))+1), axis=0, ignore_index=True).iloc[:K, :]
        color_table.reset_index(drop=True, inplace=True)

    post_umi = post.loc[:, factor_header].sum(axis = 0).astype(int).values
    post_weight = post.loc[:, factor_header].sum(axis = 0).values.astype(float)
    post_weight /= post_weight.sum()

    top_gene = []
    top_gene_neighbor = []
    # Top genes by Chi2
    de.sort_values(by=['Factor', sortby],ascending=False,inplace=True)
    de["Rank"] = de.groupby(by = "Factor")[sortby].rank(ascending=False, method = "min").astype(int)
    for k, kname in enumerate(factor_header):
        indx = de.Factor.eq(kname)
        v = de.loc[indx & ( (de.Rank < mtop) | \
                ((de.log10pval > min_log10p) & (de.FoldChange >= args.min_fc)) ), \
                'Feature'].iloc[:ntop].values
        if len(v) == 0:
            top_gene.append([kname, '.'])
        else:
            top_gene.append([kname, ', '.join(v)])
    if neighbor_de is not None:
        neighbor_de.sort_values(by=['Factor', neighbor_sortby],ascending=False,inplace=True)
        neighbor_de["Rank"] = neighbor_de.groupby(by = "Factor")[neighbor_sortby].rank(ascending=False, method = "min").astype(int)
        for k, kname in enumerate(factor_header):
            indx = neighbor_de.Factor.eq(kname)
            v = neighbor_de.loc[indx & ( (neighbor_de.Rank < mtop) | \
                    ((neighbor_de.log10pval > min_log10p) & (neighbor_de.FoldChange >= args.min_fc)) ), \
                    'Feature'].iloc[:ntop].values
            if len(v) == 0:
                top_gene_neighbor.append('.')
            else:
                top_gene_neighbor.append(', '.join(v))
    # Top genes by fold change
    de.sort_values(by=['Factor','FoldChange'],ascending=False,inplace=True)
    de["Rank"] = de.groupby(by = "Factor").FoldChange.rank(ascending=False, method = "min").astype(int)
    for k, kname in enumerate(factor_header):
        indx = de.Factor.eq(kname)
        v = de.loc[indx & ( (de.Rank < mtop) | \
                ((de.log10pval > min_log10p) & (de.FoldChange >= args.min_fc)) ), \
                'Feature'].iloc[:ntop].values
        if len(v) == 0:
            top_gene[k].append('.')
        else:
            top_gene[k].append(', '.join(v))
    # Top genes by absolute weight
    for k, kname in enumerate(factor_header):
        if post_umi[k] < 10:
            top_gene[k].append('.')
        else:
            v = post[args.feature_label].iloc[np.argsort(-post.loc[:, kname].values)[:ntop] ].values
            top_gene[k].append(', '.join(v))

    # Summary
    table = pd.DataFrame({'Factor':factor_header,
                          'RGB':color_table.RGB.values,
                        'Weight':post_weight, 'PostUMI':post_umi,
                        'TopGene_pval':[x[1] for x in top_gene],
                        'TopGene_fc':[x[2] for x in top_gene],
                        'TopGene_weight':[x[3] for x in top_gene] })
    if neighbor_de is not None:
        table.insert(table.columns.get_loc("TopGene_pval") + 1, "TopGene_specific", top_gene_neighbor)
    oheader = ["Factor", "RGB", "Weight", "PostUMI", "TopGene_pval"]
    if neighbor_de is not None:
        oheader.append("TopGene_specific")
    oheader += ["TopGene_fc", "TopGene_weight"]

    # Anchor genes used for initialization if applicable
    if os.path.exists(args.anchor):
        ak = pd.read_csv(args.anchor, sep='\t', names = ["Factor", "Anchors"], dtype={"Factor":str})
        table = table.merge(ak, on = "Factor", how = "left")
        oheader.insert(4, "Anchors")
        logging.info(f"Read anchor genes from {args.anchor}")

    if not args.keep_order:
        table.sort_values(by = 'Weight', ascending = False, inplace=True)


    if os.path.isfile(args.annotation):
        anno = {x:x for x in factor_header}
        nanno = 0
        with open(args.annotation) as f:
            for line in f:
                x = line.strip().split('\t')
                if len(x) < 2:
                    break
                anno[x[0]] = x[1]
                nanno += 1
        if nanno > 0:
            table["Factor"] = table["Factor"].map(anno)

    f = output_pref+".info.tsv"
    table.loc[table.PostUMI.ge(10), oheader].to_csv(f, sep='\t', index=False, header=True, float_format="%.5f")
    with open(f, 'r') as rf:
        lines = rf.readlines()
    header = lines[0].strip().split('\t')
    rows = [ list(enumerate(row.strip().split('\t') )) for row in lines[1:]]

    # Load template
    env = Environment(loader=FileSystemLoader(os.path.dirname(ejs)))
    template = env.get_template(os.path.basename(ejs))
    # Render the HTML file
    html_output = template.render(header=header, rows=rows, image_base64=None, tree_image_alt=None, tree_image_caption=None)

    f=output_pref+".html"
    with open(f, "w") as html_file:
        html_file.write(html_output)

    print(f)

if __name__ == "__main__":
    factor_report(sys.argv[1:])
