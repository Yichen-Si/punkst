import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.colors
from jinja2 import Environment, FileSystemLoader

def factor_report(_args):

    parser = argparse.ArgumentParser(prog="factor_report")
    parser.add_argument('--de', type=str, help='')
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
    # DE genes
    if not os.path.exists(args.de):
        sys.exit(f"Cannot find DE file")
    de = pd.read_csv(args.de, sep='\t', dtype={'factor':str})

    output_pref = args.output_pref

    factor_header = list(post.columns[1:])
    for u in factor_header:
        post[u] = post[u].astype(float)
    print(factor_header)

    color_table['RGB'] = [','.join(x) for x in np.clip((color_table.loc[:, ['R','G','B']].values).astype(int), 0, 255).astype(str) ]
    color_table['HEX'] = [ matplotlib.colors.to_hex(v) for v in np.clip(color_table.loc[:, ['R','G','B']].values / 255, 0, 1) ]

    post_umi = post.loc[:, factor_header].sum(axis = 0).astype(int).values
    post_weight = post.loc[:, factor_header].sum(axis = 0).values.astype(float)
    post_weight /= post_weight.sum()

    top_gene = []
    # Top genes by Chi2
    de.sort_values(by=['factor','Chi2'],ascending=False,inplace=True)
    de["Rank"] = de.groupby(by = "factor").Chi2.rank(ascending=False, method = "min").astype(int)
    for k, kname in enumerate(factor_header):
        indx = de.factor.eq(kname)
        v = de.loc[indx & ( (de.Rank < mtop) | \
                ((de.pval <= args.max_pval) & (de.FoldChange >= args.min_fc)) ), \
                'gene'].iloc[:ntop].values
        if len(v) == 0:
            top_gene.append([kname, '.'])
        else:
            top_gene.append([kname, ', '.join(v)])
    # Top genes by fold change
    de.sort_values(by=['factor','FoldChange'],ascending=False,inplace=True)
    de["Rank"] = de.groupby(by = "factor").FoldChange.rank(ascending=False, method = "min").astype(int)
    for k, kname in enumerate(factor_header):
        indx = de.factor.eq(kname)
        v = de.loc[indx & ( (de.Rank < mtop) | \
                ((de.pval <= args.max_pval) & (de.FoldChange >= args.min_fc)) ), \
                'gene'].iloc[:ntop].values
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
    oheader = ["Factor", "RGB", "Weight", "PostUMI", "TopGene_pval", "TopGene_fc", "TopGene_weight"]

    # Anchor genes used for initialization if applicable
    if os.path.exists(args.anchor):
        ak = pd.read_csv(args.anchor, sep='\t', names = ["Factor", "Anchors"], dtype={"Factor":str})
        table = table.merge(ak, on = "Factor", how = "left")
        oheader.insert(4, "Anchors")
        logging.info(f"Read anchor genes from {args.anchor}")

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

    f = output_pref+".factor.info.tsv"
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

    f=output_pref+".factor.info.html"
    with open(f, "w") as html_file:
        html_file.write(html_output)

    print(f)

if __name__ == "__main__":
    factor_report(sys.argv[1:])
