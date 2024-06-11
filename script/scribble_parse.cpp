#include "punkst.h"
#include "scribble.h"

// Return the position of the kth mismatch (0-based)
int32_t prefix_match(const char* seq1, const char* seq2, uint32_t l, int32_t mm = 0) {
    int32_t i = 0, m = 0;
    while (i < l) {
        if (seq1[i] != seq2[i]) {
            m++;
            if (m > mm) {
                break;
            }
        }
        i++;
    }
    if (mm == 0) {
        return i;
    }
    // In case of consecutive mismatches at the end
    while (i >= 0 && seq1[i] != seq2[i]) {
        i--;
    }
    return i + 1;
};

void skip_fq(htsFile* fq, kstring_t* str, int32_t nl = 4) {
    for (int32_t i = 0; i < nl; ++i)
        hts_getline(fq, KS_SEP_LINE, str);
};

int32_t cmdFASTQscribble(int32_t argc, char** argv) {

    std::string fq1f, fq2f, out, outf, configf;
    int32_t primer_mismatch = 1, primer_gap = 0, primer_minmatch = 10;
    int32_t statbc_mismatch = 1;
    int32_t spacer_editdist = 4;
    int32_t variable_spacer = 4;
    int32_t debug = 0, verbose = 500000;
    bool write_r1 = false;
    bool write_r1_hash = false;

    // Parse input parameters
    paramList pl;
    BEGIN_LONG_PARAMS(longParameters)
        LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("fq1", &fq1f, "")
        LONG_STRING_PARAM("fq2", &fq2f, "")
        LONG_STRING_PARAM("config", &configf, "Config file")
        LONG_INT_PARAM("primer_mismatch", &primer_mismatch, "Mismatch allowed in primer")
        LONG_INT_PARAM("primer_gap", &primer_gap, "")
        LONG_INT_PARAM("primer_minmatch", &primer_minmatch, "Minimum match in primer")
        LONG_INT_PARAM("statbc_mismatch", &statbc_mismatch, "Mismatch allowed in static barcode")
        LONG_INT_PARAM("spacer_editdist", &spacer_editdist, "Edit distance allowed in spacer")
        LONG_INT_PARAM("variable_spacer", &variable_spacer, "Variable spacer length before the upstream primer")
        LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out", &out, "")
        LONG_PARAM("write_r1", &write_r1, "Write R1")
        LONG_PARAM("write_r1_hash", &write_r1_hash, "Write R1 hash")
        LONG_INT_PARAM("debug", &debug, "Debug")
        LONG_STRING_PARAM("output", &out, "Output file")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    ScribbleConfig config(configf);

    htsFile* fq1 = hts_open(fq1f.c_str(), "r");
    htsFile* fq2 = hts_open(fq2f.c_str(), "r");
    kstring_t str1; str1.l = str1.m = 0; str1.s = NULL;
    kstring_t str2; str2.l = str2.m = 0; str2.s = NULL;
    int32_t lstr1, lstr2;
    uint64_t nread = 0, nrec = 0, nskip1 = 0, nskip2 = 0, n_sbc = 0, n_wo_sbc = 0;
    outf = out + ".tsv.gz";
    htsFile* wf = hts_open(outf.c_str(), "zw");

    // Check the distance between protospacer and static barcode
    std::stringstream ss;
    for (auto& it : config.protospacer) {
        ss << " " << seq_iupac_mismatch(it.second.c_str(), config.staticBarcode.c_str(), config.ntSpacer);
    }
    notice("Distance between protospacer and the prefix of static barcode %s:%s", config.staticBarcode.c_str(), ss.str().c_str());
    // Check the distance between protospacer and the prefix of downstream sequence
    ss.str("");
    for (auto& it : config.protospacer) {
        int32_t st1, ed1;
        int32_t score = LocalAlignmentEditDistance(config.downstreamFixed.c_str(), it.second.c_str(), config.ntSpacer+config.ntSpacer/3, config.ntSpacer, st1, ed1);
        ss << it.first << ": " << score << ", (" << st1 << ", " << ed1 << "). ";
    }
    notice("Distance between protospacer and the prefix of downstream sequence: %s", ss.str().c_str());


    // // construct a regex that matches string that contains at least one of the protospacers or the prefix the downstream sequence
    // std::string flt = "";
    // for (auto& it : config.protospacer) {
    //     flt += it.second + "|";
    // }
    // std::regex exact_spacer(flt);
    // std::regex exact_dspref(config.downstreamFixed.substr(0, config.ntSpacer));
    // // std::regex exact_primer("[ACGT]{1:4}" + config.upstreamPrimer);

    // Read the pair of fastq files, currently only use the 4N+1 lines
    while((lstr2 = hts_getline(fq2, KS_SEP_LINE, &str2)) > 0) {
        lstr1 = hts_getline(fq1, KS_SEP_LINE, &str1);
        if (lstr2 <= 0) {
            error("Unexpected EOF in FASTQ R2");
        }
        // Read the seq reads
        lstr1 = hts_getline(fq1, KS_SEP_LINE, &str1);
        lstr2 = hts_getline(fq2, KS_SEP_LINE, &str2);
        if (lstr1 <= 0 || lstr2 <= 0) {
            error("Unexpected EOF in FASTQ R1 or R2");
        }

        nread++;
        if (nread % verbose == 0) {
            notice("Processed %lu reads, %.3f do not match upstream primer; %.3f have primer but lack protospacer. Processed %d (%.3f) reads with SCRIBBLE structure.", nread, nskip1*1.0/nread, nskip2*1.0/nread, nrec, nrec*1.0/nread);
        }

        int32_t primer_offset = 0; // 1~4 starting position of upstream primer
        int32_t primer_len = 0; // length of upstream primer
        int32_t statbar_offset = 0; // starting position of static barcode
        int32_t statbar_miss; // mismatch with static barcode pattern

        // Find the longest match with prefix of upstreamPrimer
        // Starting from position 1~5, pick the longest match
        for (int32_t i = 1; i <= variable_spacer + 1; ++i) {
            int32_t j = prefix_match(str2.s + i, config.upstreamPrimer.c_str(), config.upPrimLen, primer_mismatch);
            if (j > primer_len) {
                primer_len = j;
                primer_offset = i;
                if (j == config.upPrimLen) {
                    break;
                }
            }
        }
        std::string variable_spacer = std::string(str2.s, primer_offset);
if (debug) {
    std::cout << "Primer " << primer_len << " " << primer_offset << " "
    << variable_spacer << " " << std::string(str2.s+primer_offset, primer_len) << std::endl;
}

        if (primer_len < primer_minmatch) {
            nskip1++;
            skip_fq(fq1, &str1, 2);
            skip_fq(fq2, &str2, 2);
            continue;
        }
// if (debug) {
//     std::cout << std::string(str2.s + primer_offset, primer_len) << std::endl;
// }
        // Find the static barcode
        statbar_miss = config.FindStatisBarcodeFromFront(
            str2.s, lstr2, statbar_offset,
            primer_offset + primer_len - 2, primer_offset + primer_len + primer_gap);
        int32_t scribble_st = primer_offset + primer_len; // if there is no stable barcode
        std::string statbar = ".";
        if (statbar_miss < statbc_mismatch) {
            scribble_st = statbar_offset + config.statBcLen;
            statbar = std::string(str2.s + statbar_offset, config.statBcLen);
            n_sbc++;
        } else {
            statbar_offset = primer_offset + primer_len;
        }
if (debug) {
    std::cout << "sBC " << statbar_miss << " " << statbar_offset << std::endl;
}
        // Find the first occurence of any of the protospacer
        int32_t st1 = -1, ed1 = -1;
        int32_t score = 0, min_score = lstr2;
        int32_t score_nick, score_cs;
        int32_t ptr = scribble_st;
        int32_t wsize = config.ntSpacer + config.ntBarcode + spacer_editdist;
        if (wsize > lstr2 - ptr) {
            wsize = lstr2 - ptr;
        }
        std::vector<int32_t> spacer_st = {0}, spacer_ed = {0};
        std::vector<int32_t> spacer_id = {0};
        std::stringstream spacer_score;
        for (int32_t i = 0; i < config.npair; ++i) {
            score = LocalAlignmentEditDistance(str2.s + ptr,
                    config.protospacer[config.protospacerIds[i]].c_str(),
                    wsize, config.ntSpacer, st1, ed1);
            if (score < min_score) {
                min_score = score;
                spacer_st[0] = st1 + ptr; // inclusion
                spacer_ed[0] = ed1 + ptr; // exclusion
                spacer_id[0] = i;
                if (min_score < spacer_editdist/2) {
                    break;
                }
            }
        }
        spacer_score << min_score << ";";
        if (min_score > spacer_editdist) {
            nskip2++;
            skip_fq(fq1, &str1, 2);
            skip_fq(fq2, &str2, 2);
            continue;
        }
        if (statbar_miss >= statbc_mismatch) {
            n_wo_sbc++;
        }
if (debug) {
    std::cout << "SCRIBBLE " << spacer_st[0] << std::endl;
    std::cout << "\tProtospacer " << config.protospacerIds[spacer_id[0]] << ": " << min_score << " " << spacer_st[0] << " " << spacer_ed[0]-spacer_st[0] << std::endl;
    std::cout << "\t" << std::string(str2.s+spacer_st[0], spacer_ed[0]-spacer_st[0]) << std::endl;
}

        // Find protospacer and mutable barcode pairs
        int32_t k = 0;
        std::string bc_seq = std::string();
        std::vector<int32_t> snv; // point mutations in the 4bp barcode
        std::vector<std::string> barcode_id;
        while (true) {
            // Find next protospacer
            ptr = spacer_ed[k];
            if (lstr2 - ptr < config.ntSpacer + config.ntBarcode) {
                break;
            }
            if (wsize > lstr2 - ptr) {
                wsize = lstr2 - ptr;
            }
            min_score = lstr2;
            int32_t st, ed, idx;
            for (int32_t i = 0; i < config.npair; ++i) {
                if (i == spacer_id[k]) {
                    continue;
                }
                score = LocalAlignmentEditDistance(str2.s + ptr,
                        config.protospacer[config.protospacerIds[i]].c_str(),
                        wsize, config.ntSpacer, st1, ed1);
                if (score < min_score) {
                    min_score = score;
                    st = st1 + ptr;
                    ed = ed1 + ptr;
                    idx = i;
                    if (min_score < 2) {
                        break;
                    }
                }
            }
            if (min_score > spacer_editdist) {
                break;
            }
            // ad hoc creteria to avoid false positive
            if (min_score > 1) {
                if (config.spacerTail[config.protospacerIds[spacer_id[k]]].compare(0, config.ntEditDelete, str2.s+spacer_ed[k], config.ntEditDelete) == 0) {
                    break;
                }
                if (min_score > 2 && st - spacer_ed[k] != config.ntBarcode) {
                    break;
                }
            }
            k += 1;
            spacer_st.push_back(st);
            spacer_ed.push_back(ed);
            spacer_id.push_back(idx);
            spacer_score << min_score << ";";

if (debug) {
    std::cout << "\tProtospacer " << config.protospacerIds[spacer_id[k]] << ": " << min_score << " " << spacer_st[k] << " " << spacer_ed[k]-spacer_st[k] << std::endl;
    std::cout << "\t" << std::string(str2.s+spacer_st[k], spacer_ed[k]-spacer_st[k]) << std::endl;
}

            // Identify the mutable barcode
            std::string bc;
            int32_t bcl = spacer_st[k] - spacer_ed[k-1];

// if (bcl != config.ntBarcode) {
// std::cout << "Length between two spacer differs from expected\n";
// std::cout << config.protospacerIds[spacer_id[k-1]] << ", " << config.protospacerIds[spacer_id[k]] << ". " << min_score << std::endl;
// std::cout << std::string(str2.s+spacer_st[k-1], spacer_ed[k-1]-spacer_st[k-1]) << " " << spacer_ed[k-1]-spacer_st[k-1] << std::endl;
// std::cout << std::string(str2.s+spacer_ed[k-1], spacer_st[k]-spacer_ed[k-1]) << " " << bcl << std::endl;
// std::cout << std::string(str2.s+spacer_st[k], spacer_ed[k]-spacer_st[k]) << " " << spacer_ed[k]-spacer_st[k] << std::endl;
// }

            if (bcl < config.ntBarcode) {
                snv.push_back(-1);
                bc_seq += std::string(str2.s+ptr, bcl) + ";";
                barcode_id.push_back(".");
                continue;
            }
            int32_t min_score_bc = config.ntBarcode;
            int32_t bc_st = spacer_ed[k-1];
            for (auto & v : config.validPair[config.protospacerIds[spacer_id[k]]] ) {
                std::string& u = config.mutableBarcode[v];
                for (ptr = spacer_ed[k-1]; ptr <= spacer_st[k] - config.ntBarcode; ptr++) {
                    int32_t m = 0;
                    for (int32_t i = 0; i < config.ntBarcode; ++i) {
                        if (str2.s[ptr + i] != u[i]) {
                            m++;
                        }
                    }
                    if (m < min_score_bc) {
                        bc_st = ptr;
                        min_score_bc = m;
                        bc = v;
                    }
                }
            }
            bc_seq += std::string(str2.s+bc_st, config.ntBarcode) + ";";
            snv.push_back(min_score_bc);
            barcode_id.push_back(bc);
if (debug) {
    std::cout << "\tBC " << min_score_bc << " " << bc << std::endl;
}
        }

        // match with downstream sequence
        ptr = spacer_ed[k];
        wsize = config.downFixLen;
        if (lstr2 - ptr < config.downFixLen) {
            wsize = lstr2 - ptr;
        }
        if (wsize < 1) {
            score = -1;
            score_nick = -1;
            score_cs = -1;
        } else {
            score = LocalAlignmentEditDistance(str2.s + ptr, config.downstreamFixed.c_str(), lstr2 - ptr, wsize, st1, ed1);
            wsize = config.nickFixLen;
            score_nick = LocalAlignmentEditDistance(str2.s + ptr, config.nickingFixed.c_str(), lstr2 - ptr, wsize, st1, ed1);
            wsize = config.capSeqLen;
            score_cs = LocalAlignmentEditDistance(str2.s + ptr, config.captureSequence.c_str(), lstr2 - ptr, wsize, st1, ed1);
        }

if (debug) {
    std::cout << "Downstream " << score << ", " << score_nick << ", " << score_cs << " " << ptr << std::endl;
}

std::stringstream landmark;
landmark << primer_offset << "," << statbar_offset;
for (auto & v : spacer_st) {
    landmark << "," << v;
}

// output: match length in primer, Static barcode, scribble pairs, spacer list, barcode list, barcode mutations counts, barcode seq, distance to downstream sequence

hprintf(wf, "%d\t%s\t%d\t", primer_len, statbar.c_str(), k);
for (auto& v : spacer_id) {
    hprintf(wf, "%s;", config.protospacerIds[v].c_str());
}
hprintf(wf, "\t");
for (auto& v : barcode_id) {
    hprintf(wf, "%s;", v.c_str());
}
hprintf(wf, "\t");
for (auto& v : snv) {
    hprintf(wf, "%d;", v);
}
hprintf(wf, "\t%s\t%s\t%d\t%d\t%d", bc_seq.c_str(), spacer_score.str().c_str(), score, score_nick, score_cs);
if (write_r1_hash) {
    uint64_t r1_hash = seq2nt5(str1.s, std::min(lstr1, MAX_NT5_UNIT_64));
    hprintf(wf, "\t%lu", r1_hash);
}
if (write_r1) {
    hprintf(wf, "\t%s", str1.s);
}
hprintf(wf, "\t%s\t%s\n", variable_spacer.c_str(), landmark.str().c_str() );

        nrec++;

if (debug > 0 && nrec > debug) {
    break;
}
        skip_fq(fq1, &str1, 2);
        skip_fq(fq2, &str2, 2);
    }



// std::cout << "upstreamPrimer: " << config.upstreamPrimer << " " << config.upstreamPrimer.size() << std::endl;
// std::cout << "staticBarcode: " << config.staticBarcode << " " << config.staticBarcode.size() << std::endl;
// std::cout << "config.downstreamFixed: " << config.downstreamFixed << " " << config.downstreamFixed.size() << std::endl;
// std::cout << "nickingSite: " << config.nickingFixed << " " << config.nickingFixed.size() << std::endl;
// std::cout << "config.captureSequence: " << config.captureSequence << " " << config.captureSequence.size() << std::endl;
// std::cout << "editDeletion: " << config.ntEditDelete << std::endl;
// for (auto& it : config.validPair) {
//     std::cout << it.first << " " << config.protospacer[it.first] << std::endl;
//     for (auto v : it.second) {
//         std::cout << v << ": " << config.mutableBarcode[v] << std::endl;
//     }
// }

    hts_close(fq1);
    hts_close(fq2);
    free(str1.s);
    free(str2.s);
    hts_close(wf);

    outf = out + ".log";
    wf = hts_open(outf.c_str(), "w");
    hprintf(wf, "Total reads: %lu\n", nread);
    hprintf(wf, "Skipped due to primer mismatch: %lu\n", nskip1);
    hprintf(wf, "Skipped due to missing the first protospacer: %lu\n", nskip2);
    hprintf(wf, "Reads with proper static barcode: %lu\n", n_sbc);
    hprintf(wf, "Reads without proper static barcode but with protospacer: %lu\n", n_wo_sbc);
    hprintf(wf, "Reads with SCRIBBLE structure: %lu\n", nrec);
    hts_close(wf);

    return 0;
}
