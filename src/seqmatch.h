#ifndef __SEQ_MATCH_H
#define __SEQ_MATCH_H

#include "punkst.h"
#include "qgenlib/seq_utils.h"
#include <unordered_map>

// int32_t seq2bits_expand(const char* seq, int32_t len, std::vector<uint64_t>& realization, int32_t max_ambig = 1);

template <typename T>
class BcdMatch {

private:

int32_t max_ambig = 1;
int32_t max_mismatch = 1;
bool ambig_processed = false;
std::unordered_map<uint64_t, T> sbcd_ref;
std::unordered_map<uint64_t, T> ambig_sbcd;
std::unordered_map<uint64_t, std::pair<uint16_t, T> > ambig_temp;
std::vector<std::unordered_map<uint64_t, std::vector<uint64_t> > > pattern2sbcd;

void build_patterns();
int32_t query_unambig_besthit_(uint64_t bcd, uint64_t& cb, T& val, int32_t _max_nm);

public:

uint32_t bcd_len, kmer_size, npattern;
bool exact_only, allow_ambig_ref, allow_ambig_query;
std::vector<uint64_t> patterns;

BcdMatch() {}
BcdMatch(uint32_t _bcd_len, uint32_t _k, bool _exact = false, bool _ar = true, bool _aq = true, int32_t _max_nm = 1);
void init(uint32_t _bcd_len, uint32_t _k, bool _exact = false, bool _ar = true, bool _aq = true, int32_t _max_nm = 1);
void reset();
void process_ambig_ref();
void add_keys(uint64_t bcd);
int32_t add_ref(const char* seq, T& val);
int32_t query(const char* seq, uint64_t& cb, T& val, int32_t _max_nm = -1);
size_t index_size(std::vector<size_t>& pattern_sizes);

};

#include "seqmatch.hpp"

#endif // __SEQ_MATCH_H
