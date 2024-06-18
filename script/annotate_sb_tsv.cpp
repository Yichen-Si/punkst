#include "punkst.h"
#include "seqmatch.h"
#include "qgenlib/tsv_reader.h"
#include "qgenlib/qgen_utils.h"
#include "htslib/hfile.h"

int32_t tsv_next(htsFile* rf, kstring_t& str, std::vector<std::string>& fields) {
    int32_t lstr = hts_getline(rf, KS_SEP_LINE, &str);
    if (lstr <= 0) {return 0;}
    split(fields, "\t", str.s, str.l + 1, true, false, true);
    return lstr;
}

int32_t cmdTsvAnnoSb(int32_t argc, char** argv) {

	std::string intsv, output;
	std::vector<std::string> sbcd_list;
	int32_t bcd_len = -1, kmer_size = 16, max_mismatch = 1;
    int32_t icol_q = -1, icol_s = 0, icol_x = 3, icol_y = 4;
	int32_t verbose = 500000, debug = 0;
	bool exact_only = false, allow_1N_ref = false, allow_1N_query = false;
	std::string outmode("wz");
    bool check_consecutive = false;
    bool query_unique = false;
    bool reverse_complement = false;

	// Parse input parameters
	paramList pl;

	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
		LONG_STRING_PARAM("input", &intsv, "Input TSV file")
		LONG_MULTI_STRING_PARAM("sbcd", &sbcd_list, "Input TSV file with spatial barcode info")
		LONG_INT_PARAM("bcd-len", &bcd_len, "Spatial barcode length")
		LONG_INT_PARAM("kmer", &kmer_size, "K-mer size")
        LONG_INT_PARAM("col-query", &icol_q, "Column index for barcode sequence in the query tsv")
		LONG_INT_PARAM("col-seq", &icol_s, "Column index for barcode sequence in the reference tsv")
		LONG_INT_PARAM("col-x", &icol_x, "Column index for x")
		LONG_INT_PARAM("col-y", &icol_y, "Column index for y")
		LONG_PARAM("exact-only", &exact_only, "Only exact matches")
		LONG_PARAM("allow-1N-ref", &allow_1N_ref, "Allow one N in reference")
		LONG_PARAM("allow-1N-query", &allow_1N_query, "Allow one N in query")
        LONG_PARAM("check-consecutive", &check_consecutive, "Query TSV is sorted by barcode")
        LONG_PARAM("reverse-complement", &reverse_complement, "Query and reference barcodes are reverse complements")
        LONG_PARAM("query-unique", &query_unique, "Query TSV has unique barcode")
		LONG_PARAM_GROUP("Output Options", NULL)
		LONG_STRING_PARAM("output", &output, "Output file")
		LONG_STRING_PARAM("output-mode", &outmode, "Output mode")
		LONG_INT_PARAM("verbose", &verbose, "Verbose")
		LONG_INT_PARAM("debug", &debug, "Debug")
	END_LONG_PARAMS();

	pl.Add(new longParams("Available Options", longParameters));
	pl.Read(argc, argv);
	pl.Status();

	if (intsv.empty() || output.empty() || sbcd_list.size() < 1 || bcd_len < 0 || icol_q < 0) {
		error("--intput, --output, --sbcd, --bcd-len, --col-query are required but at least one is missing");
	}
    if (query_unique) {
        check_consecutive = false;
    }

    htsFormat fmt={unknown_category,text_format,{-1,-1},no_compression,-1,NULL};
    htsFile* rf = hts_open_format(intsv.c_str(), "r", &fmt);
    if (rf == NULL) {
        error("Failed to open input file %s", intsv.c_str());
    }
    kstring_t str;
    str.l = str.m = 0; str.s = NULL;
    int32_t lstr = hts_getline(rf, KS_SEP_LINE, &str);
    std::vector<std::string> fields;
    split(fields, "\t", str.s, lstr+1, true, false, true);
    int32_t nfields = fields.size();
    if (icol_q >= nfields) {
        error("Column index is out of range in file %s (%d, %d)", intsv.c_str(), icol_q, nfields);
    }
    int32_t bcd_len_query =  fields[icol_q].length();
    if (bcd_len > bcd_len_query) {
        error("--bcd-len larger longer than the length of barcode sequence in the input TSV file %s", intsv.c_str());
    }

	BcdMatch<uint64_t> bcd_match(bcd_len, kmer_size, exact_only, allow_1N_ref, allow_1N_query, max_mismatch);

	uint64_t ref_tot = 0, ref_ambig = 0, ref_skip = 0;
	for (auto& intsv : sbcd_list) {
		tsv_reader sbcd_tsv(intsv.c_str());
		sbcd_tsv.read_line();
		if (icol_s >= sbcd_tsv.nfields || icol_x >= sbcd_tsv.nfields || icol_y >= sbcd_tsv.nfields) {
			error("Column index is out of range in file %s", intsv.c_str());
		}
		const char* seq = sbcd_tsv.str_field_at(icol_s);
		if (bcd_len > strlen(seq)) {
			error("--bcd-len larger longer than the length of barcode sequence in the input TSV file %s", intsv.c_str());
		}
		while (true) {
			ref_tot++;
			if (ref_tot % verbose == 0) {
				notice("Processed %lu reference barcodes, %lu contain 1 ambiguous base, %lu are skipped", ref_tot, ref_ambig, ref_skip);
			}
			uint64_t val = (uint64_t) sbcd_tsv.int_field_at(icol_x) << 32 | sbcd_tsv.int_field_at(icol_y);
			int32_t ret = bcd_match.add_ref(sbcd_tsv.str_field_at(icol_s), val);
			if (ret < 0) {
				ref_skip++;
			} else if (ret > 0) {
				ref_ambig++;
			}
			if (!sbcd_tsv.read_line()) {
				break;
			}
			if (debug % 3 == 1 && ref_tot > debug) {
				break;
			}
		}
	}
	bcd_match.process_ambig_ref();
    notice("Processed %lu reference barcodes, %lu contain 1 ambiguous base, %lu are skipped", ref_tot, ref_ambig, ref_skip);

    uint64_t n_rec = 0, n_match = 0, n_exact = 0, n_missing = 0, n_lazy = 0, n_query = 0;
	uint64_t prev_cb, prev_xy;
	int32_t prev_n_mismatch = -1;
	char prev_bcd_org_str[bcd_len+1] = {'.'};
    htsFile* wf = hts_open(output.c_str(), outmode.c_str());
    while (lstr > 0) {
        n_rec++;
        if (n_rec % verbose == 0) {
            notice("Processed %lu records, queried %lu times, %lu matched, %lu exact, %lu missing, %lu skipped, %lu lazy", n_rec, n_query, n_match, n_exact, n_missing, n_lazy);
        }
        if (debug && n_rec > debug) {
            break;
        }
        char* seq = (char*) fields[icol_q].c_str();
        if (reverse_complement) {
            seq_revcomp(seq, bcd_len_query);
        }
        uint64_t cb, xy;
        int32_t n_mismatch;
        if (check_consecutive) {
            if (strncmp(seq, prev_bcd_org_str, bcd_len) == 0) {
                n_lazy++;
                if (prev_n_mismatch < 0) {
                    lstr = tsv_next(rf, str, fields);
                    continue;
                }
                n_mismatch = prev_n_mismatch;
                cb = prev_cb;
                xy = prev_xy;
            } else {
                n_query++;
                strncpy(prev_bcd_org_str, seq, bcd_len);
                n_mismatch = bcd_match.query(seq, cb, xy);
                prev_n_mismatch = n_mismatch;
                if (n_mismatch < 0) {
                    lstr = tsv_next(rf, str, fields);
                    continue;
                }
                prev_cb = cb;
                prev_xy = xy;
            }
        } else {
            n_query++;
            n_mismatch = bcd_match.query(seq, cb, xy);
            if (n_mismatch < 0) {
                lstr = tsv_next(rf, str, fields);
                continue;
            }
        }
        n_match++;
        if (n_mismatch == 0) {
            n_exact++;
        }
        std::stringstream ss;
        ss << n_mismatch << "\t" << (xy >> 32) << "\t" << (xy & 0xFFFFFFFF);
        hprintf(wf, "%.*s\t%s\n", lstr, str.s, ss.str().c_str());

        lstr = tsv_next(rf, str, fields);
    }
    hts_close(rf);
    hts_close(wf);

    return 0;
}
