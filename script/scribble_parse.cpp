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
    return i;
};

void skip_fq(htsFile* fq, kstring_t* str, int32_t nl = 4) {
    for (int32_t i = 0; i < nl; ++i)
        hts_getline(fq, KS_SEP_LINE, str);
};

int32_t cmdFASTQscribble(int32_t argc, char** argv) {

    // std::string fq1f, fq2f;
    std::string out, outf, configf;
    std::vector<std::string> fq1vec, fq2vec;
    int32_t primer_mismatch = 1, primer_minmatch = -1, primer_gap = 0;
    int32_t statbc_mismatch = 1;
    int32_t spacer_editdist = 4;
    int32_t barcode_indelnt = 1;
    int32_t debug = 0, verbose = 500000;
    int32_t last_proto_minmatch = 10;
    int32_t signature_trim_read_end = 5;
    bool allow_missing_static_barcode = false;
    bool write_r1 = false;
    bool write_r1_hash = false;

    // Parse input parameters
    paramList pl;
    BEGIN_LONG_PARAMS(longParameters)
        LONG_PARAM_GROUP("Input options", NULL)
        LONG_MULTI_STRING_PARAM("fq1", &fq1vec, "")
        LONG_MULTI_STRING_PARAM("fq2", &fq2vec, "")
        LONG_STRING_PARAM("config", &configf, "Config file")
        LONG_INT_PARAM("primer_mismatch", &primer_mismatch, "Mismatch allowed in primer (before reaching --primer_minmatc)")
        LONG_INT_PARAM("primer_minmatch", &primer_minmatch, "Minimum match in primer")
        LONG_INT_PARAM("primer_gap", &primer_gap, "Maximum allowed gap between upstream primer and static barcode")
        LONG_INT_PARAM("statbc_mismatch", &statbc_mismatch, "Mismatch allowed in static barcode")
        LONG_INT_PARAM("spacer_editdist", &spacer_editdist, "Edit distance allowed in spacer")
        LONG_INT_PARAM("barcode_indelnt", &barcode_indelnt, "Indel length allowed in the mutable barcode")
        LONG_INT_PARAM("signature_trim_read_end", &signature_trim_read_end, "When recording subsequence as identifier, ignore the last N bases")
        LONG_PARAM("allow_missing_static_barcode", &allow_missing_static_barcode, "Allow missing static barcode")
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
    if (primer_minmatch < 0) {
        primer_minmatch = config.upPrimLen;
    }
    std::cout << "Read config file\n";
    std::cout << "upstreamPrimer: " << config.upstreamPrimer << " " << config.upstreamPrimer.size() << std::endl;
    std::cout << "staticBarcode: " << config.staticBarcode << " " << config.staticBarcode.size() << std::endl;
    std::cout << "variableSpacer " << config.variableSpacer_min << ", " << config.variableSpacer_max << std::endl;
    std::cout << "validPair\n";
    for (auto& it : config.validPair) {
        std::cout << it.first << " " << config.protospacer[it.first] << std::endl;
        for (auto v : it.second) {
            std::cout << v << ": " << config.mutableBarcode[v] << std::endl;
        }
    }

    kstring_t str1; str1.l = str1.m = 0; str1.s = NULL;
    kstring_t str2; str2.l = str2.m = 0; str2.s = NULL;
    int32_t lstr1, lstr2;
    uint64_t nread = 0, nrec = 0, n_sbc = 0, n_wo_sbc = 0;
    uint64_t nskip1 = 0, nskip2 = 0;
    outf = out + ".tsv.gz";
    htsFile* wf = hts_open(outf.c_str(), "wz");

    // Check the distance between protospacer and static barcode
    int32_t proto_prefix_to_statbc_dist = config.statBcLen;
    std::stringstream ss;
    for (auto& it : config.protospacer) {
        int32_t d = seq_iupac_mismatch(it.second.c_str(), config.staticBarcode.c_str(), config.ntSpacer);
        if (d < proto_prefix_to_statbc_dist) {
            proto_prefix_to_statbc_dist = d;
        }
        ss << " " << d;
    }
    notice("Distance between static barcode and the prefix of protospacer %s:%s", config.staticBarcode.c_str(), ss.str().c_str());
    // Check the distance between protospacer and the prefix of downstream sequence
    ss.str("");
    for (auto& it : config.protospacer) {
        int32_t st1, ed1, st1_match;
        int32_t score = LocalAlignmentEditDistance(config.downstreamFixed.c_str(), it.second.c_str(), config.ntSpacer+config.ntSpacer/3, config.ntSpacer, st1, ed1, st1_match);
        ss << it.first << ": " << score << ", (" << st1 << ", " << ed1 << "). ";
    }
    notice("Distance between the protospacer and the prefix of downstream sequence: %s", ss.str().c_str());

    for (size_t fqi = 0; fqi < fq1vec.size(); ++fqi) {

    htsFile* fq1 = hts_open(fq1vec[fqi].c_str(), "r");
    htsFile* fq2 = hts_open(fq2vec[fqi].c_str(), "r");
    // Read the pair of fastq files, currently only use the 4N+1 lines
    while((lstr2 = hts_getline(fq2, KS_SEP_LINE, &str2)) > 0) {
        lstr1 = hts_getline(fq1, KS_SEP_LINE, &str1);
        if (lstr1 <= 0) {
            error("Unexpected EOF in FASTQ R1");
        }
        // Read the seq reads
        lstr1 = hts_getline(fq1, KS_SEP_LINE, &str1);
        lstr2 = hts_getline(fq2, KS_SEP_LINE, &str2);
        if (lstr1 <= 0 || lstr2 <= 0) {
            error("Unexpected EOF in FASTQ R1 or R2");
        }

        nread++;
        if (nread % verbose == 0) {
            notice("Processed %lu reads, %.3f do not match upstream primer; %.3f have primer but lack protospacer; %.3f have protospacer but do not have static barcode. Processed %d (%.3f) reads with SCRIBBLE structure.", nread, nskip1*1.0/nread, nskip2*1.0/nread, n_wo_sbc*1./nread, nrec, nrec*1.0/nread);
        }

        int32_t primer_offset = 0; // 0-based, Start position of upstream primer
        int32_t primer_len = 0; // prefix length of primer before max mismatch
        // Pick the starting position that gives the longest match
        // with the prefix of upstreamPrimer
        for (int32_t i = config.variableSpacer_min; i <= config.variableSpacer_max; ++i) {
            int32_t j = prefix_match(str2.s + i, config.upstreamPrimer.c_str(), config.upPrimLen, primer_mismatch);
            if (j > primer_len) {
                primer_len = j;
                primer_offset = i;
                if (j == config.upPrimLen) {
                    break;
                }
            }
        }

        if (primer_len < primer_minmatch) {
            nskip1++;
            skip_fq(fq1, &str1, 2);
            skip_fq(fq2, &str2, 2);
            continue;
        }

        // Find the first occurence of any of the protospacer
        int32_t st1 = -1, ed1 = -1, st1_match = -1;
        int32_t score = 0, min_score = lstr2;
        int32_t ptr = primer_offset + primer_len;
        if (lstr2 - ptr < config.ntSpacer) { // should never happen
            nskip2++;
            skip_fq(fq1, &str1, 2);
            skip_fq(fq2, &str2, 2);
            continue;
        }
        int32_t wsize = config.statBcLen + config.ntSpacer + spacer_editdist;
        if (wsize > lstr2 - ptr) {
            wsize = lstr2 - ptr;
        }
        std::vector<int32_t> spacer_st = {0}, spacer_ed = {0};
        std::vector<int32_t> spacer_mst = {0};
        std::vector<int32_t> spacer_id = {0};
        for (int32_t i = 0; i < config.npair; ++i) {
            std::string& proto = config.protospacer[config.protospacerIds[i]];
            score = config.ntSpacer - prefix_match(str2.s+ptr, proto.c_str(), config.ntSpacer, spacer_editdist);
            if (score > 0) {
                score = LocalAlignmentEditDistance(str2.s+ptr, proto.c_str(),
                        wsize, config.ntSpacer, st1, ed1, st1_match);
            } else {
                st1 = 0;
                ed1 = config.ntSpacer;
                st1_match = 0;
                while(st1_match < config.ntSpacer && str2.s[ptr+st1_match] != proto[st1_match]) {
                    st1_match++;
                }
            }
            if (score < min_score) {
                min_score = score;
                spacer_st[0] = st1 + ptr; // inclusion
                spacer_ed[0] = ed1 + ptr; // exclusion
                spacer_mst[0] = st1_match + ptr;
                spacer_id[0] = i;
                if (min_score < spacer_editdist) {
                    break;
                }
            }
        }
        if (min_score > spacer_editdist) {
            nskip2++;
            skip_fq(fq1, &str1, 2);
            skip_fq(fq2, &str2, 2);
            continue;
        }
        std::stringstream spacer_score;
        spacer_score << min_score << ";";
if (debug % 3 == 1) {
std::cout << "SCRIBBLE " << spacer_st[0] << "," << spacer_mst[0] << std::endl;
std::cout << "\tProtospacer " << config.protospacerIds[spacer_id[0]] << ": " << min_score << " " << spacer_st[0] << "," << spacer_mst[0] << " " << spacer_ed[0]-spacer_st[0] << std::endl;
}

        // Go back to check stable barcode
        int32_t statbar_offset = -1; // starting position of static barcode
        std::string statbar = ".";
        n_wo_sbc++;
        if (spacer_mst[0] - (primer_offset + primer_len) >= config.statBcLen) {
            // May contain static barcode
            int32_t statbar_miss = config.FindStatisBarcodeFromFront(
                str2.s, lstr2, statbar_offset,
                primer_offset + primer_len, // min offset
                primer_offset + config.upPrimLen + primer_gap);
if (debug % 3 == 1) {
std::cout << "\tsBC " << statbar_miss << " " << statbar_offset << " (" << primer_offset + primer_len << ", " << primer_offset + config.upPrimLen + primer_gap << ")" << std::endl;
}
            if (statbar_miss < statbc_mismatch) {
                statbar = std::string(str2.s + statbar_offset, config.statBcLen);
                n_sbc++;
                n_wo_sbc--;
            } else {
                statbar_offset = -1;
            }
        }
        if (statbar_offset < 0 && !allow_missing_static_barcode) {
            skip_fq(fq1, &str1, 2);
            skip_fq(fq2, &str2, 2);
            continue;
        }

        // Find subsequent protospacer and mutable barcode pairs
        int32_t k = 0;
        std::string bc_seq = std::string();
        std::vector<int32_t> snv; // point mutations in the 4bp barcode
        std::vector<std::string> barcode_id;
        wsize = config.ntBarcode + config.ntSpacer + spacer_editdist;
        while (true) {
            if (config.spacerTail[config.protospacerIds[spacer_id[k]]].compare(0, config.ntEditDelete, str2.s+spacer_ed[k], config.ntEditDelete) == 0) {
                break; // Seen a full protospacer
            }
            // Find next protospacer
            ptr = spacer_ed[k];
            if (lstr2 - ptr < config.ntBarcode + last_proto_minmatch) {
                break;
            }
            if (wsize > lstr2 - ptr) {
                wsize = lstr2 - ptr;
            }
            min_score = lstr2;
            int32_t idx;
            for (int32_t i = 0; i < config.npair; ++i) {
                if (i == spacer_id[k]) {
                    continue;
                }
                std::string& proto = config.protospacer[config.protospacerIds[i]];
                score = config.ntSpacer - prefix_match(str2.s+ptr+config.ntBarcode, proto.c_str(), config.ntSpacer, spacer_editdist);
                if (score != 0) {
                    score = LocalAlignmentEditDistance(str2.s + ptr, proto.c_str(), wsize, config.ntSpacer, st1, ed1, st1_match);
                } else {
                    st1 = config.ntBarcode;
                    ed1 = config.ntBarcode + config.ntSpacer;
                    st1_match = 0;
                    while(st1_match < config.ntSpacer && str2.s[ptr+st1+st1_match] != proto[st1_match]) {
                        st1_match++;
                    }
                    st1_match += st1;
                }
                if (score < min_score) {
                    min_score = score;
                    idx = i;
                    if (min_score < spacer_editdist) {
                        break;
                    }
                }
            }
            if (min_score > spacer_editdist) {
                break;
            }
            k += 1;
            spacer_st.push_back(st1+ptr);
            spacer_ed.push_back(ed1+ptr);
            spacer_mst.push_back(st1_match+ptr);
            spacer_id.push_back(idx);
            spacer_score << min_score << ";";

if (debug % 3 == 1) {
std::cout << "\tProtospacer " << config.protospacerIds[spacer_id[k]] << ": " << min_score << " " << spacer_st[k] << "," << spacer_mst[k] << " " << spacer_ed[k]-spacer_st[k] << std::endl;
}
            // Identify the mutable barcode
            std::string bc;
            int32_t bcl = spacer_mst[k] - spacer_ed[k-1];

if (debug % 3 == 1 && bcl != config.ntBarcode) {
std::cout << "Length between two spacer differs from expected\n";
std::cout << config.protospacerIds[spacer_id[k-1]] << ", " << config.protospacerIds[spacer_id[k]] << ". " << min_score << std::endl;
std::cout << std::string(str2.s+spacer_ed[k-1], spacer_st[k]-spacer_ed[k-1]) << " " << bcl << std::endl;
std::cout << std::string(str2.s+spacer_st[k], spacer_ed[k]-spacer_st[k]) << " " << spacer_ed[k]-spacer_st[k] << std::endl;
}

            int32_t bc_st = spacer_ed[k-1];
            int32_t bc_st_max = spacer_mst[k] - config.ntBarcode + barcode_indelnt;
            if (bc_st_max < bc_st) {
                snv.push_back(-1);
                bc_seq += std::string(str2.s+ptr, bcl) + ";";
                barcode_id.push_back(".");
                continue;
            }
            int32_t min_score_bc = config.ntBarcode + 1;
            for (auto & v : config.validPair[config.protospacerIds[spacer_id[k]]] ) {
                std::string& u = config.mutableBarcode[v];
                for (ptr = spacer_ed[k-1]; ptr <= bc_st_max; ptr++) {
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
if (debug % 3 == 1) {
    std::cout << "\tBC " << bc << " " << min_score_bc << std::endl;
}
        }

        // match with downstream sequence
        int32_t last_fixed_base = spacer_ed.back(); // find the first variable 3'utr
        int32_t score_nick, score_cs;
        ptr = spacer_ed[k];
        if (lstr2 - ptr < std::min(config.minTail, 10)) {
            score = -1;
            score_nick = -1;
            score_cs = -1;
        } else {
            wsize = std::min(lstr2 - ptr, config.downFixLen);
            score = LocalAlignmentEditDistance(str2.s + ptr, config.downstreamFixed.c_str(), lstr2 - ptr, wsize, st1, ed1, st1_match);
            if (score < config.downFixLen * 0.2 && ed1 > last_fixed_base) {
                last_fixed_base = ed1;
            }

            wsize = std::min(lstr2 - ptr, config.downFixLen);
            score_nick = LocalAlignmentEditDistance(str2.s + ptr, config.nickingFixed.c_str(), lstr2 - ptr, wsize, st1, ed1, st1_match);
            if (score_nick < config.nickFixLen * 0.2 && ed1 > last_fixed_base) {
                last_fixed_base = ed1;
            }

            wsize = std::min(lstr2 - ptr, config.capSeqLen);
            score_cs = LocalAlignmentEditDistance(str2.s + ptr, config.captureSequence.c_str(), lstr2 - ptr, wsize, st1, ed1, st1_match);
            if (score_cs < config.capSeqLen * 0.2 && ed1 > last_fixed_base) {
                last_fixed_base = ed1;
            }
        }

if (debug % 3 == 1) {
    std::cout << "\nDownstream " << score << "," << score_nick << "," << score_cs << " " << ptr << std::endl;
}

// temporary - in case there is no static barcode,
// write the first few bases after the last fixed downstream sequences and
// the last few bases of the read, hope they can be surrogate identifier
// this is arbitrary, but as long as it is deterministic
uint64_t sig_seq = 0;
if (last_fixed_base < lstr2 - signature_trim_read_end) {
    int32_t l = std::min(32, lstr2 - signature_trim_read_end - last_fixed_base);
    sig_seq = seq2bits(str2.s + last_fixed_base, l, 0) << 32 | seq2bits(str2.s + lstr2 - l, l, 0);
}

std::stringstream landmark;
landmark << primer_offset << "," << statbar_offset;
for (auto & v : spacer_mst) {
    landmark << "," << v;
}
landmark << "\t" << last_fixed_base << "\t" << sig_seq;

        // output: variable sapcer, static barcode, match length in primer,
        // number of scribble pairs, spacer list, spacer edits,
        // barcode list, barcode mutations counts, barcode seq,
        // edit distance to downstream sequences
        // (Optional): read 1 sequence or hash of (part of) it
        // landmark positions (positions of the first matches to protospacers)
        // position of the last base (+1) after the fixed downstream seqs
        // hash of some pieces of the reamining sequences
        std::string variable_spacer = std::string(str2.s, primer_offset);
        if (primer_offset == 0) {
            variable_spacer = ".";
        }
        hprintf(wf, "%s\t%s\t%d\t%d\t", variable_spacer.c_str(), statbar.c_str(), primer_len, k);
        for (auto& v : spacer_id) {
            hprintf(wf, "%s;", config.protospacerIds[v].c_str());
        }
        hprintf(wf, "\t%s\t", spacer_score.str().c_str());
        if (barcode_id.size() == 0) {
            hprintf(wf, ".");
            bc_seq = ".";
        } else {
            for (auto& v : barcode_id) {
                hprintf(wf, "%s;", v.c_str());
            }
        }
        hprintf(wf, "\t");
        if (snv.size() == 0) {
            hprintf(wf, ".");
        } else {
            for (auto& v : snv) {
                hprintf(wf, "%d;", v);
            }
        }
        hprintf(wf, "\t%s\t%d\t%d\t%d\t%s", bc_seq.c_str(), score, score_nick, score_cs, landmark.str().c_str());
        if (write_r1_hash) {
            uint64_t r1_hash = seq2nt5(str1.s, std::min(lstr1, MAX_NT5_UNIT_64));
            hprintf(wf, "\t%lu", r1_hash);
        }
        if (write_r1) {
            hprintf(wf, "\t%s", str1.s);
        }
        hprintf(wf, "\n");

        nrec++;

if (debug && nrec > debug) {
    break;
}
        skip_fq(fq1, &str1, 2);
        skip_fq(fq2, &str2, 2);
    }

    hts_close(fq1);
    hts_close(fq2);
    }


    free(str1.s);
    free(str2.s);
    hts_close(wf);

    notice("Finished. Processed %lu reads, %.3f do not match upstream primer; %.3f have primer but lack protospacer; %.3f have protospacer but do not have static barcode. Processed %d (%.3f) reads with SCRIBBLE structure.", nread, nskip1*1.0/nread, nskip2*1.0/nread, n_wo_sbc*1./nread, nrec, nrec*1.0/nread);

    outf = out + ".log";
    wf = hts_open(outf.c_str(), "w");
    hprintf(wf, "Total reads: %lu\n", nread);
    hprintf(wf, "Skipped due to primer missing or mismatch: %lu\n", nskip1);
    hprintf(wf, "Skipped due to missing the first protospacer: %lu\n", nskip2);
    hprintf(wf, "Reads with proper static barcode: %lu\n", n_sbc);
    if (allow_missing_static_barcode)
        hprintf(wf, "Reads that have protospacer(s) but without proper static barcode: %lu\n", n_wo_sbc);
    hprintf(wf, "Reads with SCRIBBLE structure: %lu\n", nrec);
    hts_close(wf);

    return 0;
}
