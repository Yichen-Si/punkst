#include "seq_utils.h"
#include "qgenlib/qgen_error.h"
#include <cstdio>
#include <cstring>

/* constant table - from Heng Li at
   https://github.com/lh3/seqtk/blob/master/seqtk.c
   https://github.com/samtools/htslib/blob/develop/hts.c */

// ASCII to IUPAC mapper
const unsigned char seq_nt16_table[256] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 /*'-'*/, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 1, 14, 2, 13, 15, 15, 4, 11, 15, 15, 12, 15, 3, 15, 15,
    15, 15, 5, 6, 8, 15, 7, 9, 0, 10, 15, 15, 15, 15, 15, 15,
    15, 1, 14, 2, 13, 15, 15, 4, 11, 15, 15, 12, 15, 3, 15, 15,
    15, 15, 5, 6, 8, 15, 7, 9, 0, 10, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};

// ASCII to XACGTN representation
const unsigned char seq_nt6_table[256] = {
    0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 1, 5, 2, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 1, 5, 2, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,

    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

// ASCII to ACGNT representation
const unsigned char seq_nt5_table[256] = {
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 0, 3, 1, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 0, 3, 1, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,

    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};

// IUPAC16 to letter mapping
const char *seq_nt16_rev_table = "XACMGRSVTWYHKDBN";

// ACGNT to ASCII conversion 
const char *seq_nt5_rev_table = "ACGNT";

// Force-convert IUPAC16 to AGCT(01234)
const unsigned char seq_nt16to4_table[] = {4, 0, 1, 4, 2, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4};

// 4-bit comparison string between IUPAC16 to ACGT
// A -> 1000, C -> 0100 M=AC -> 1010
const unsigned char seq_nt16comp_table[] = {0, 8, 4, 12, 2, 10, 9, 14, 1, 6, 5, 13, 3, 11, 7, 15};

// How many letters does a IUPACT code represent?
const int bitcnt_table[] = {4, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

// reverse complement table
const char comp_tab[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 'T', 'V', 'G', 'H', 'E', 'F', 'C', 'D', 'I', 'J', 'M', 'L', 'K', 'N', 'O',
    'P', 'Q', 'Y', 'S', 'A', 'A', 'B', 'W', 'X', 'R', 'Z', 91, 92, 93, 94, 95,
    64, 't', 'v', 'g', 'h', 'e', 'f', 'c', 'd', 'i', 'j', 'm', 'l', 'k', 'n', 'o',
    'p', 'q', 'y', 's', 'a', 'a', 'b', 'w', 'x', 'r', 'z', 123, 124, 125, 126, 127};

// reverse complement of string sequences
void seq_revcomp(char *seq, int32_t l)
{
  int32_t i, c0, c1;
  for (i = 0; i < l >> 1; ++i)
  { // reverse complement sequence
    c0 = comp_tab[(int32_t)seq[i]];
    c1 = comp_tab[(int32_t)seq[l - 1 - i]];
    seq[i] = c1;
    seq[l - 1 - i] = c0;
  }
  if (l & 1) // complement the remaining base
    seq[l >> 1] = comp_tab[(int)seq[l >> 1]];
}

// count the number of mismatches compared to IUPAC pattern
int32_t seq_iupac_mismatch(const char *seq, const char *pattern, int32_t len)
{
  int32_t mismatches = 0;
  unsigned char iupac, nt6;
  for (int32_t i = 0; i < len; ++i)
  {
    if (pattern[i] != 'N')
    {
      iupac = seq_nt16_table[pattern[i]];
      nt6 = seq_nt6_table[seq[i]];
      if (nt6 > 0 && nt6 < 5)
      {
        if ((seq_nt16comp_table[iupac] & (0x01 << (4 - nt6))) == 0)
          ++mismatches;
      }
      else
      {
        ++mismatches;
      }
    }
  }
  return mismatches;
}

// convert sequences into 2bit strings (len <= 32)
uint64_t seq2bits(const char *seq, int32_t len, uint8_t nonACGTs)
{
  uint64_t bits = 0;
  uint8_t c;
  for (int32_t i = 0; i < len; ++i)
  {
    c = seq_nt6_table[seq[i]];
    if (c == 0 || c == 5)
      c = nonACGTs + 1;
    bits = ((bits << 2) | ((c - 1) & 0x03));
  }
  return bits;
}

// convert sequences into 2bit strings (len <= 64)
uint128_t seq2bits_128(const char *seq, int32_t len, uint8_t nonACGTs)
{
  uint128_t bits = 0;
  uint8_t c;
  for (int32_t i = 0; i < len; ++i)
  {
    c = seq_nt6_table[seq[i]];
    if (c == 0 || c == 5)
      c = nonACGTs + 1;
    bits = ((bits << 2) | ((c - 1) & 0x03));
  }
  return bits;
}

// convert sequences into base-5 binaries (len <= 27)
uint64_t seq2nt5(const char *seq, int32_t len)
{
  uint64_t bits = 0;
  for (int32_t i = 0; i < len; ++i)
  {
    bits = bits * 5 + (int32_t)seq_nt5_table[seq[i]];
  }
  // error("%s %llu", seq, bits);
  return bits;
}

// convert sequences into base-5 binaries (len <= 55)
uint128_t seq2nt5_128(const char *seq, int32_t len)
{
  uint128_t bits = 0;
  for (int32_t i = 0; i < len; ++i)
  {
    bits = bits * 5 + (int32_t)seq_nt5_table[seq[i]];
  }
  // error("%s %llu", seq, bits);
  return bits;
}

// convert base-5 binaries into sequences (len <= 27)
bool nt52seq(uint64_t nt5, int32_t len, char *seq) {
  int32_t i;
  for (i = len - 1; i >= 0; --i) {
    seq[i] = seq_nt5_rev_table[nt5 % 5];
    nt5 /= 5;
  }
  seq[len] = '\0';
  return nt5 == 0; // if nt5 is not zero, then the sequence is too long
}

// convert base-5 binaries into sequences (len <= 55)
bool nt52seq_128(uint128_t nt5, int32_t len, char *seq) {
  int32_t i;
  for (i = len - 1; i >= 0; --i) {
    seq[i] = seq_nt5_rev_table[nt5 % 5];
    nt5 /= 5;
  }
  seq[len] = '\0';
  return nt5 == 0; // if nt5 is not zero, then the sequence is too long
}

// convert long sequences into multiple chunks of base-5 binaries (len <= 27)
int32_t seq2nt5multi(const char *seq, int32_t lseq, uint64_t *nt5s, int32_t nt5unit) {
  // determine the number of chunks
  int32_t nchunks = (lseq + nt5unit - 1) / nt5unit;
  int32_t offset = 0;
  int32_t i;
  for (i = 0; i < nchunks; ++i) {
    nt5s[i] = seq2nt5(seq + i * nt5unit, offset + nt5unit > lseq ? lseq - offset : nt5unit);
    offset += nt5unit;
  }
  return nchunks;
}

// convert multiple chunks of base-5 binaries into sequences
bool nt5multi2seq(uint64_t *nt5s, int32_t lseq, char *seq, int32_t nt5unit) {
  int32_t nchunks = (lseq + nt5unit - 1) / nt5unit;
  int32_t offset = 0;
  int32_t i;
  for (i = 0; i < nchunks; ++i) {
    if (!nt52seq(nt5s[i], offset + nt5unit > lseq ? lseq - offset : nt5unit, seq + i * nt5unit))
      return false;
    offset += nt5unit;
  }
  seq[lseq] = '\0';
  return true;
}
