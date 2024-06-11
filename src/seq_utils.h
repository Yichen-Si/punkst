#ifndef __SEQ_UTILS_H
#define __SEQ_UTILS_H

#include <cstdint>
#include <cstddef>
#include <functional>

#define MAX_NT5_UNIT_64 27
#define MAX_NT5_UNIT_128 55

// code to support 128-bit integer
typedef __uint128_t uint128_t; // 128-bit unsigned integer
struct uint128_hash {
    std::size_t operator()(const uint128_t& key) const {
        return std::hash<uint64_t>()(key >> 64) ^ std::hash<uint64_t>()(key);
    }
};

int32_t seq_iupac_mismatch(const char *seq, const char *pattern, int32_t len);
void seq_revcomp(char *seq, int32_t len);

uint64_t seq2bits(const char *seq, int32_t len, uint8_t nonACGTs = 0x03);

uint64_t seq2nt5(const char *seq, int32_t len);
bool nt52seq(uint64_t nt5, int32_t len, char *seq);
int32_t seq2nt5multi(const char *seq, int32_t lseq, uint64_t *nt5s, int32_t nt5unit = 27);
bool nt5multi2seq(uint64_t *nt5s, int32_t lseq, char *seq, int32_t nt5unit = 27);

uint128_t seq2bits_128(const char *seq, int32_t len, uint8_t nonACGTs = 0x03);

uint128_t seq2nt5_128(const char *seq, int32_t len);
bool nt52seq_128(uint128_t nt5, int32_t len, char *seq);

#endif // __SEQ_UTILS_H
