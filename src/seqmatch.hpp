#ifndef __SEQ_MATCH_HPP
#define __SEQ_MATCH_HPP

#include "seqmatch.h"

template <typename T>
BcdMatch<T>::BcdMatch(uint32_t _bcd_len, uint32_t _k, bool _exact, bool _ar, bool _aq, int32_t _max_nm) {
    init(_bcd_len, _k, _exact, _ar, _aq, _max_nm);
}

template <typename T>
void BcdMatch<T>::init(uint32_t _bcd_len, uint32_t _k, bool _exact, bool _ar, bool _aq, int32_t _max_nm) {
    reset();
    bcd_len = _bcd_len;
    kmer_size = _k;
    exact_only = _exact;
    allow_ambig_ref = _ar;
    allow_ambig_query = _aq;
    max_mismatch = _max_nm;
    if (bcd_len > 32) {
        std::cerr << "BcdMatch::init: Currently only support barcodes with length <= 32\n";
        bcd_len = 32;
    }
    if (kmer_size >= bcd_len) {
        std::cerr << "BcdMatch::init: kmer size >= barcode length, will allow only exact match\n";
        exact_only = true;
        kmer_size = bcd_len;
    }
    if (exact_only) {
        npattern=0;
        max_mismatch = 0;
        allow_ambig_ref = false;
        allow_ambig_query = false;
    } else {
        uint32_t nspace = bcd_len - kmer_size;
        npattern = bcd_len / nspace;
        if (bcd_len % nspace > 0) {++npattern;}
        build_patterns();
    }
}

template <typename T>
void BcdMatch<T>::reset() {
    patterns.clear();
    pattern2sbcd.clear();
    sbcd_ref.clear();
    ambig_sbcd.clear();
    ambig_temp.clear();
    ambig_processed = false;
}

template <typename T>
void BcdMatch<T>::build_patterns() {
    for (uint32_t i = 0; i < npattern; ++i) {
        uint64_t mask = 0xFFFFFFFFFFFFFFFF;
        for (uint32_t j = 0; j < bcd_len; ++j) {
            if ((j + i) % npattern == 0) {
                mask &= ~(3ULL << j * 2);
            }
        }
        patterns.push_back(mask);
    }
    pattern2sbcd.resize(npattern);
}


template <typename T>
void BcdMatch<T>::add_keys(uint64_t bcd) {
    for (uint32_t i = 0; i < npattern; ++i) {
        uint64_t key = bcd & patterns[i];
        auto ptr = pattern2sbcd[i].find(key);
        if (ptr == pattern2sbcd[i].end()) {
            pattern2sbcd[i].emplace(key, std::vector<uint64_t>(1, bcd));
        } else {
            ptr->second.push_back(bcd);
        }
    }
}


template <typename T>
int32_t BcdMatch<T>::add_ref(const char* seq, T& val) {
    std::vector<uint8_t> nonACGTs;
    uint64_t bcd = seq2bits2(seq, bcd_len, nonACGTs);
    if (nonACGTs.size() == 0) {
        sbcd_ref[bcd] = val;
        if (!exact_only) {add_keys(bcd);}
        return 0;
    }
    // Assume reference barcodes are unique
    if (allow_ambig_ref && nonACGTs.size() <= max_ambig) {
        uint16_t abase = ((uint16_t) nonACGTs[0] << 8) | seq[nonACGTs[0]];
        ambig_temp.emplace(bcd, std::make_pair(abase, val));
        return nonACGTs.size();
    }
    return -1;
}


template <typename T>
int32_t BcdMatch<T>::query_unambig_besthit_(uint64_t bcd, uint64_t& cb, T& val, int32_t _max_nm) {
    int32_t max_nm = (_max_nm < 0) ? max_mismatch : _max_nm;
    std::unordered_map<uint64_t, int32_t> hits;
    for (uint32_t i = 0; i < npattern; ++i) {
        uint64_t key = bcd & patterns[i];
        auto ptr = pattern2sbcd[i].find(key);
        if (ptr == pattern2sbcd[i].end()) {continue;}
        for (auto ref : ptr->second) {
            int32_t nmiss = nt4_hamming_dist(bcd, ref, max_nm);
            if (nmiss <= max_nm) {
                hits.emplace(ref, nmiss);
                if (max_nm == 1 && hits.size() > 1) {
                    return -3; // multiple matching with same distance
                }
            }
        }
    }
    if (hits.size() == 0) {
        return -1;
    }
    if (hits.size() == 1) {
        cb = hits.begin()->first;
        val = sbcd_ref[cb];
        return hits.begin()->second;
    }
    int32_t min_mismatch = max_nm + 1;
    std::vector<uint64_t> min_keys;
    for (auto kv : hits) {
        if (kv.second < min_mismatch) {
            min_mismatch = kv.second;
            min_keys.clear();
            min_keys.push_back(kv.first);
        } else if (kv.second > max_nm) {
            continue;
        } else if (kv.second == min_mismatch) {
            min_keys.push_back(kv.first);
        }
    }
    if (min_keys.size() == 1) {
        cb = min_keys[0];
        val = sbcd_ref[cb];
        return min_mismatch;
    }
    return -1;
}

template <typename T>
void BcdMatch<T>::process_ambig_ref() {
    if (!allow_ambig_ref) {
        return;
    }
    int32_t nambig = 0, nkept = 0;
    for (auto kv : ambig_temp) {
        // if (kv.second.first & 0x8000) {
        //     continue;
        // }
        nambig++;
        std::vector<uint64_t> realization;
        uint8_t nt16comp = seq_nt16comp_table[seq_nt16_table[kv.second.first & 0xFF]];
        uint16_t apos = kv.second.first >> 8;
        bool collision = false;
        for (uint32_t j = 0; j < 4; ++j) {
            if (nt16comp & (1 << j)) {
                uint64_t key = kv.first | ((3ULL - j) << (bcd_len - apos - 1) * 2);
                if (sbcd_ref.find(key) == sbcd_ref.end()) {
                    realization.push_back(key);
                } else {
                    collision = true;
                    break;
                }
            }
        }
        if (collision) {
            continue;
        }
        nkept++;
        // Assume we allow at most one ambiguous nucleotide in a barcode
        // Then expanded barcodes are unique
        for (auto key : realization) {
            ambig_sbcd.emplace(key, kv.second.second);
        }
    }
    fprintf(stderr, "Processed %d ambiguous reference barcodes, kept %d that do not collide with unambiguous barcodes, expanded to %lu keys for exact match\n", nambig, nkept, ambig_sbcd.size());
    ambig_temp.clear();
    ambig_processed = true;
}


template <typename T>
int32_t BcdMatch<T>::query(const char* seq, uint64_t& cb, T& val, int32_t _max_nm) {
    if (allow_ambig_ref && !ambig_processed) {
        process_ambig_ref();
    }
    int32_t max_nm = (_max_nm < 0) ? max_mismatch : _max_nm;
    std::vector<uint8_t> nonACGTs;
    uint64_t bcd = seq2bits2(seq, bcd_len, nonACGTs);
    if ((nonACGTs.size() > 0 && !allow_ambig_query) || (nonACGTs.size() > max_ambig)) {
        return -1;
    }
    if (nonACGTs.size() == 0) {
        auto ptr = sbcd_ref.find(bcd);
        if (ptr != sbcd_ref.end()) {
            cb = bcd;
            val = ptr->second;
            return 0;
        }
        if (exact_only) {
            return -1;
        }
        if (allow_ambig_ref && max_nm >= max_ambig) {
            ptr = ambig_sbcd.find(bcd);
            if (ptr != ambig_sbcd.end()) {
                cb = bcd;
                val = ptr->second;
                return max_ambig;
            }
        }
        return query_unambig_besthit_(bcd, cb, val, max_nm);
    }
    std::vector<uint64_t> realization;
    uint8_t apos = nonACGTs[0];
    uint8_t nt16comp = seq_nt16comp_table[seq_nt16_table[seq[apos]]];
    bool hit = false;
    for (uint32_t j = 0; j < 4; ++j) {
        if (nt16comp & (1 << j)) {
            uint64_t key = bcd | ((3ULL - j) << (bcd_len - apos - 1) * 2);
            auto ptr = sbcd_ref.find(key);
            if (ptr != sbcd_ref.end()) {
                if (hit) {
                    return -1;
                }
                cb = key;
                val = ptr->second;
                hit = true;
            }
        }
    }
    if (hit) {
        return 1;
    }
    return -1;
}

template <typename T>
size_t BcdMatch<T>::index_size(std::vector<size_t>& pattern_sizes) {
    pattern_sizes.clear();
    for (uint32_t i = 0; i < npattern; ++i) {
        pattern_sizes.push_back(pattern2sbcd[i].size());
    }
    return sbcd_ref.size();
}


#endif // __SEQ_MATCH_HPP
