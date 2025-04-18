import sys, os, re, argparse
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import *
from joblib import Parallel, delayed

def gen_even_slices(n, n_packs):
    """Generate approximately even-sized slices of indices."""
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            yield np.arange(start, end)
            start = end

def log10_chi2_sf(x, df=1):
    return -np.log10(scipy.stats.chi2.sf(x, df))

def process_chunk(gene_chunk, header, total_k, total_umi):
    """Process a chunk of genes for all factors."""
    results = []

    for k, kname in enumerate(header):
        if total_k[k] <= 0:
            continue

        k_counts = gene_chunk[kname].values
        gene_tots = gene_chunk['gene_tot'].values
        gene_names = gene_chunk.index.values

        # Filter genes with counts > 0 for this factor
        valid_idx = k_counts > 0
        if not np.any(valid_idx):
            continue

        valid_k_counts = k_counts[valid_idx]
        valid_gene_tots = gene_tots[valid_idx]
        valid_names = gene_names[valid_idx]

        # Calculate contingency table values for all genes at once
        a = valid_k_counts  # cell 0,0
        b = valid_gene_tots - a  # cell 0,1
        c = total_k[k] - a  # cell 1,0
        d = total_umi - total_k[k] - valid_gene_tots + a  # cell 1,1

        # Calculate fold change
        fold_changes = (a / total_k[k]) / (b / (total_umi - total_k[k]))

        # Filter for fold change > 1
        fc_valid = fold_changes >= 1
        if not np.any(fc_valid):
            continue

        # Apply filters
        a, b, c, d = a[fc_valid], b[fc_valid], c[fc_valid], d[fc_valid]
        fold_changes = fold_changes[fc_valid]
        names = valid_names[fc_valid]
        gene_tots = valid_gene_tots[fc_valid]

        # Calculate chi-square statistic and p-value for each gene
        for i in range(len(names)):
            # Add 1 to avoid zeros (Laplace smoothing)
            tab = np.array([[a[i] + 1, b[i] + 1], [c[i] + 1, d[i] + 1]])
            chi2, p, _, _ = scipy.stats.chi2_contingency(tab, correction=False)
            results.append([names[i], kname, chi2, p, fold_changes[i], gene_tots[i]])

    return results

def de_bulk(_args):
    parser = argparse.ArgumentParser(prog="de_bulk")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--feature', type=str, default='', help='')
    parser.add_argument('--feature_label', type=str, default="gene", help='')
    parser.add_argument('--min_ct_per_feature', default=50, type=int, help='')
    parser.add_argument('--max_pval_output', default=1e-3, type=float, help='')
    parser.add_argument('--min_fold_output', default=1.5, type=float, help='')
    parser.add_argument('--min_output_per_factor', default=10, type=int, help='Even when there are no significant DE genes, output top genes for each factor')
    parser.add_argument('--thread', default=1, type=int, help='')
    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return

    pcut = args.max_pval_output
    fcut = args.min_fold_output

    # Load gene filtering set if provided
    gene_kept = set()
    if os.path.exists(args.feature):
        feature = pd.read_csv(args.feature, sep='\t', header=0)
        gene_kept = set(feature[args.feature_label].values)

    # Read aggregated count table
    info = pd.read_csv(args.input, sep='\t', header=0)
    header = [x for x in info.columns if x != args.feature_label]

    K = len(header)
    M = info.shape[0]
    info.rename(columns={args.feature_label, "gene"}, inplace=True)
    print(f"Read posterior count over {M} genes and {K} factors")

    # Apply gene filtering
    if len(gene_kept) > 0:
        info = info.loc[info.gene.isin(gene_kept), :]

    # Calculate totals once for efficiency
    info["gene_tot"] = info[header].sum(axis=1)
    info = info[info["gene_tot"] > args.min_ct_per_feature]
    info.index = info.gene.values

    # Pre-calculate these values once
    total_umi = info.gene_tot.sum()
    total_k = info[header].sum(axis=0).values

    M = info.shape[0]
    print(f"Testing {M} genes over {K} factors")

    # Prepare data for parallel processing
    results = []

    if args.thread > 1:
        # Split genes into chunks for parallel processing
        gene_chunks = []
        for idx in gen_even_slices(M, args.thread):
            gene_chunks.append(info.iloc[idx])

        # Process chunks in parallel
        with Parallel(n_jobs=args.thread, verbose=0) as parallel:
            chunk_results = parallel(delayed(process_chunk)(
                chunk, header, total_k, total_umi) for chunk in gene_chunks)

        # Combine results
        for chunk_result in chunk_results:
            results.extend(chunk_result)
    else:
        # Single-threaded processing
        results = process_chunk(info, header, total_k, total_umi)

    # Create and process results dataframe
    if results:
        chidf = pd.DataFrame(results, columns=['gene', 'factor', 'Chi2', 'pval', 'FoldChange', 'gene_total'])

        # Rank genes within each factor
        chidf["Rank"] = chidf.groupby("factor")["Chi2"].rank(ascending=False)

        # Filter results
        chidf = chidf.loc[
            ((chidf.pval < pcut) & (chidf.FoldChange > fcut)) |
            (chidf.Rank <= args.min_output_per_factor)
        ]

        # Sort by factor and Chi2 score
        chidf.sort_values(by=['factor', 'Chi2'], ascending=[True, False], inplace=True)

        # Format output columns
        chidf.Chi2 = chidf.Chi2.map(lambda x: "{:.1f}".format(x))
        chidf.FoldChange = chidf.FoldChange.map(lambda x: "{:.2f}".format(x))
        chidf.gene_total = chidf.gene_total.astype(int)

        # Calculate log10 p-values efficiently
        chidf["log10pval"] = np.array([log10_chi2_sf(float(x)) for x in chidf.Chi2])

        # Drop temporary columns
        chidf.drop(columns='Rank', inplace=True)

        # Write results to file
        chidf.to_csv(args.output, sep='\t', float_format="%.2e", index=False)
    else:
        print("No significant results found.")

if __name__ == "__main__":
    de_bulk(sys.argv[1:])
