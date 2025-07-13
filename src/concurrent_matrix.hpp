#pragma once

#include <array>
#include <vector>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>
#include <algorithm>
#include <iostream>

// Return the lowest set bit of x
constexpr inline uint64_t lowbit(uint64_t x) noexcept { return x & (~x + 1); }

// Bit-scan-reverse: index of the most-significant set bit (0-based)
constexpr inline int bsr(uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(x);
#else
    int pos = -1;
    while (x) { x >>= 1u; ++pos; }
    return pos;
#endif
}

inline uint64_t safe_left_shift(int shift) {
    if (shift >= 63) throw std::overflow_error("ConcurrentColMatrix: shift overflow");
    return uint64_t{1} << shift;
}

// Column-major matrix
template<class T>
class ConcurrentColMatrix {
    static_assert(std::is_trivially_copyable<T>::value,
                  "ConcurrentColMatrix requires POD / trivially-copyable T");
    static_assert(sizeof(size_t) >= 8, "ConcurrentColMatrix assumes a 64-bit platform");

    using Segment = std::unique_ptr<std::atomic<T>[]>;

public:
    ConcurrentColMatrix(size_t num_rows, int base_column_shift = 3)
        : num_rows_(num_rows), row_capacity_(num_rows + 1),
          base_column_shift_(base_column_shift), s_(sizeof(std::atomic<T>)) {
        num_               = 1;
        num_columns_       = 0;
        column_capacity_   = uint64_t{1} << base_column_shift_;
        const uint64_t seg_elems = column_capacity_ * row_capacity_;
        data_[0].reset(new std::atomic<T>[seg_elems]);
        std::memset(data_[0].get(), 0, s_ * seg_elems);
    }

    ConcurrentColMatrix(ConcurrentColMatrix&& other) noexcept
        : num_(other.num_), row_capacity_(other.row_capacity_),
          base_column_shift_(other.base_column_shift_),
          num_rows_(other.num_rows_), num_columns_(other.num_columns_),
          column_capacity_(other.column_capacity_),
          data_(std::move(other.data_)), s_(other.s_) {
        other.num_ = 0; other.num_columns_ = other.column_capacity_ = 0;
    }
    ConcurrentColMatrix& operator=(ConcurrentColMatrix&&) noexcept = default;
    ConcurrentColMatrix(const ConcurrentColMatrix&)            = delete;
    ConcurrentColMatrix& operator=(const ConcurrentColMatrix&) = delete;

    size_t GetC()                   const noexcept { return num_columns_; }
    size_t GetR()                   const noexcept { return num_rows_;    }

    T Get(size_t r, size_t c)       const noexcept {
        return At(r,c).load(std::memory_order_relaxed);}
    T GetSum(size_t c)              const noexcept {
        return At(num_rows_, c).load(std::memory_order_relaxed);}

    void Set(size_t r, size_t c, T v)      noexcept {
        At(r,c).store(v);}
    void SetSum(size_t c, T v)             noexcept {
        At(num_rows_, c).store(v);}

    void Inc(size_t r, size_t c, T v = 1)  noexcept {
        At(r,c)          += v;
        At(num_rows_, c) += v;
    }
    void Dec(size_t r, size_t c, T v = 1)  noexcept {
        At(r,c)          -= v;
        At(num_rows_, c) -= v;
    }

    void Clear() noexcept {
        for (int seg = 0; seg < num_; ++seg) {
            const uint64_t seg_cols = uint64_t{1} << (base_column_shift_ + seg);
            const uint64_t seg_elems = seg_cols * row_capacity_;
            std::memset(data_[seg].get(), 0, s_ * seg_elems);
        }
    }

    void ClearCol(size_t c) noexcept {
        if (c >= num_columns_) return;
        At(num_rows_, c).store(0, std::memory_order_relaxed);
        std::memset(&At(0,c), 0, s_ * row_capacity_);
    }

    void Grow(size_t new_ncol) {
        if (new_ncol <= num_columns_) return;

        std::lock_guard<std::mutex> guard(mutex_);

        while (new_ncol > column_capacity_) {
            const uint64_t seg_cols = uint64_t{1} << (base_column_shift_ + num_);
            const uint64_t seg_elems = seg_cols * row_capacity_;

            std::cout << "ConcurrentColMatrix: Growing from "
                        << column_capacity_ << " to ";
            data_[num_].reset(new std::atomic<T>[seg_elems]);
            std::memset(data_[num_].get(), 0, s_ * seg_elems);
            column_capacity_ += seg_cols;
            ++num_;
            std::cout << column_capacity_ << " columns.\n";
        }
        num_columns_ = new_ncol;
    }

    // merge segments into one
    void Consolidate() {
        if (num_ == 1) return;
        std::lock_guard<std::mutex> guard(mutex_);

        // round capacity to next power-of-two
        while (column_capacity_ != lowbit(column_capacity_))
            column_capacity_ += lowbit(column_capacity_);

        const uint64_t seg_elems = column_capacity_ * row_capacity_;
        auto* new_data = new std::atomic<T>[seg_elems];
        std::memset(new_data, 0, s_ * seg_elems);

        size_t col_cursor = 0;
        for (int seg = 0; seg < num_; ++seg) {
            const uint64_t seg_cols = uint64_t{1} << (base_column_shift_ + seg);
            const uint64_t seg_bytes = seg_cols * s_ * row_capacity_;
            for (size_t col = 0; col < seg_cols; ++col) {
                std::memcpy(new_data + (col_cursor + col) * row_capacity_,
                            data_[seg].get() + col * row_capacity_,
                            s_ * row_capacity_);
            }
            col_cursor += seg_cols;
        }

        data_[0].reset(new_data);
        base_column_shift_ = bsr(column_capacity_);

        for (int i = 1; i < 64; ++i) data_[i].reset(nullptr);
        num_ = 1;

        std::cout << "ConcurrentColMatrix: Consolidated into one segment ("
                  << column_capacity_ << " columns, 2^"
                  << base_column_shift_ << ").\n";
    }

    // copy a column
    template<class InputIt>
    void SetCol(size_t c, InputIt first, InputIt last) noexcept {
        const size_t n = static_cast<size_t>(std::distance(first,last));
        if (c >= num_columns_ || n == 0) return;
        auto* base = &At(0,c); // pointer to row 0
        size_t i   = 0;
        T colsum = 0;
        for (; first != last && i < num_rows_; ++first, ++i) {
            base[i].store(static_cast<T>(*first), std::memory_order_relaxed);
            colsum += static_cast<T>(*first);
        }
        for (; i < num_rows_; ++i)
            base[i].store(0, std::memory_order_relaxed);
        At(num_rows_, c).store(colsum, std::memory_order_relaxed);
    }
    void SetCol(size_t c, const T* src) noexcept {
        if (c >= num_columns_) return;
        std::memcpy(&At(0,c), src, s_ * num_rows_);
        T colsum = 0;
        for (size_t i = 0; i < num_rows_; ++i)
            colsum += src[i];
        At(num_rows_, c).store(colsum, std::memory_order_relaxed);
    }

    std::atomic<T>* ColPtrAtomic(std::size_t c) noexcept {
        if (c >= num_columns_) return nullptr; // out of bounds
        int bucket; std::size_t bucket_c;
        find_bucket(c, bucket, bucket_c);
        return data_[bucket].get() + bucket_c * row_capacity_;
    }
    const std::atomic<T>* ColPtrAtomic(std::size_t c) const noexcept {
        if (c >= num_columns_) return nullptr; // out of bounds
        int bucket; std::size_t bucket_c;
        find_bucket(c, bucket, bucket_c);
        return data_[bucket].get() + bucket_c * row_capacity_;
    }

    void Reset(size_t new_ncol) {
        std::lock_guard<std::mutex> guard(mutex_);
        const uint64_t first_seg_cols = uint64_t{1} << base_column_shift_;
        const uint64_t required_cols  = new_ncol;
        // keep current first segment if it's large enough
        if (first_seg_cols >= required_cols) {
            for (int i = 1; i < 64; ++i) data_[i].reset(nullptr);
            num_             = 1;
            num_columns_     = new_ncol;
            column_capacity_ = first_seg_cols;
            std::memset(data_[0].get(), 0, s_ * first_seg_cols * row_capacity_);
            return;
        }
        // otherwise allocate next power-of-two ≥ required_cols
        int new_shift = base_column_shift_;
        while ((uint64_t{1} << new_shift) < required_cols) ++new_shift;
        const uint64_t new_cols   = uint64_t{1} << new_shift;
        const uint64_t seg_elems  = new_cols * row_capacity_;
        auto* new_data            = new std::atomic<T>[seg_elems];
        std::memset(new_data, 0, s_ * seg_elems);
        data_[0].reset(new_data);
        for (int i = 1; i < 64; ++i) data_[i].reset(nullptr);
        base_column_shift_ = new_shift;
        column_capacity_   = new_cols;
        num_columns_       = new_ncol;
        num_               = 1;
    }

private:
    /* ------------------------------------------------------------------
     *  Address calculation helpers
     * ---------------------------------------------------------------- */
    inline void find_bucket(size_t c, int& bucket, size_t& bucket_c) const noexcept {
        if (c < (uint64_t{1} << base_column_shift_)) {
            bucket   = 0;
            bucket_c = c;
        } else {
            bucket   = bsr((c >> base_column_shift_) + 1);
            bucket_c = c - ((uint64_t{1} << bucket) - 1) *
                           (uint64_t{1} << base_column_shift_);
        }
    }

    std::atomic<T>& At(size_t r, size_t c) const noexcept {
        int    bucket;
        size_t bucket_c;
        find_bucket(c, bucket, bucket_c);
        return data_[bucket][bucket_c * row_capacity_ + r];
    }

    /* ------------------------------------------------------------------
     *  data members
     * ---------------------------------------------------------------- */
    int                          num_ = 0; // # segments
    const size_t                 row_capacity_; // rows+1
    int                          base_column_shift_;
    size_t                       num_rows_;
    size_t                       num_columns_ = 0; // columns used
    uint64_t                     column_capacity_; // allocated columns
    std::array<Segment, 64>      data_{}; // up to 64 segments
    mutable std::mutex           mutex_;  // guards layout ops
    const size_t                 s_;      // sizeof(std::atomic<T>)
};


// A row-wise atomic matrix dynamic by column
template <class T>
class ConcurrentMatrix {
    static_assert(std::is_trivially_copyable<T>::value,
                  "ConcurrentMatrix requires POD / trivially-copyable T");
    static_assert(sizeof(size_t) >= 8, "ConcurrentMatrix assumes a 64-bit platform");

    using Segment = std::unique_ptr<std::atomic<T>[]>;

public:
    ConcurrentMatrix(size_t num_rows, int base_column_shift = 7)
        : num_rows_(num_rows), base_column_shift_(base_column_shift) {
        num_ = 1;
        num_columns_ = 0;
        column_capacity_ = uint64_t{1} << base_column_shift; // first segment
        // Round capacity of rows to next power-of-two.
        uint64_t row_capacity = num_rows_ + 1;
        while (row_capacity != lowbit(row_capacity))
            row_capacity += lowbit(row_capacity);
        row_shift_ = bsr(row_capacity);
        s_ = sizeof(std::atomic<T>);

        const uint64_t capacity = safe_left_shift(base_column_shift_ + row_shift_);
        data_[0].reset(new std::atomic<T>[capacity]);
        std::memset(data_[0].get(), 0, s_ * capacity);
    }

    // Move-constructor
    ConcurrentMatrix(ConcurrentMatrix&& other) noexcept :
            num_(other.num_), row_shift_(other.row_shift_),
            base_column_shift_(other.base_column_shift_),
            num_rows_(other.num_rows_), num_columns_(other.num_columns_),
            column_capacity_(other.column_capacity_),
            data_(std::move(other.data_)) {
        other.num_ = 0;
        other.num_columns_ = 0;
        other.column_capacity_ = 0;
    }
    ConcurrentMatrix& operator=(ConcurrentMatrix&&) noexcept = default;
    ConcurrentMatrix(const ConcurrentMatrix&)            = delete;
    ConcurrentMatrix& operator=(const ConcurrentMatrix&) = delete;

    void Fill(T value = 0) noexcept {
        for (int i = 0; i < num_; ++i) {
            const auto seg_size = safe_left_shift(base_column_shift_ + row_shift_ + i);
            std::memset(data_[i].get(), value, s_ * seg_size);
        }
    }

    void ClearRow(size_t r) {
        if (r >= num_rows_) {
            throw std::out_of_range("ConcurrentMatrix: row index out of range");
        }
        std::lock_guard<std::mutex> guard(mutex_);
        // set row r to zero
        for (int i = 0; i < num_; ++i) {
            size_t rowsize = uint64_t{1} << (base_column_shift_ + i);
            size_t offset = r * rowsize;
            std::memset(data_[i].get() + offset, 0, s_ * rowsize);
        }
    }

    void Reset(size_t new_ncol) {
        std::lock_guard<std::mutex> guard(mutex_);
        const uint64_t first_seg_cols = uint64_t{1} << base_column_shift_;
        const size_t   n_rows_plus1   = num_rows_ + 1;
        if (first_seg_cols >= new_ncol) {
            // keep first segment, drop the rest
            for (int i = 1; i < 64; ++i) data_[i].reset(nullptr);
            num_            = 1;
            num_columns_    = new_ncol;
            column_capacity_= first_seg_cols;
            std::memset(data_[0].get(), 0, s_ * first_seg_cols * n_rows_plus1);
            return;
        }

        int new_shift = base_column_shift_;
        while ((uint64_t{1} << new_shift) < new_ncol) ++new_shift;
        column_capacity_ = uint64_t{1} << new_shift;

        const uint64_t seg_elems = safe_left_shift(row_shift_ + new_shift);
        auto* new_data = new std::atomic<T>[seg_elems];
        std::memset(new_data, 0, s_ * seg_elems);
        data_[0].reset(new_data);
        for (int i = 1; i < 64; ++i) data_[i].reset(nullptr);

        base_column_shift_ = new_shift;
        num_columns_       = new_ncol;
        num_               = 1;
    }

    size_t GetC() const noexcept {
        return num_columns_;
    }
    T Get(size_t r, size_t c) const noexcept {
        return At(r, c).load(std::memory_order_relaxed);
    }
    T GetSum(size_t c) const noexcept {
        return At(num_rows_, c).load(std::memory_order_relaxed);
    }
    void Set(size_t r, size_t c, T value) noexcept {
        At(r, c).store(value);
    }
    void SetSum(size_t c, T value) noexcept {
        At(num_rows_, c).store(value);
    }
    void Inc(size_t r, size_t c, T value = 1) {
        At(r, c) += value;
        At(num_rows_, c) += value;
    }
    void Dec(size_t r, size_t c, T value = 1) {
        At(r, c) -= value;
        At(num_rows_, c) -= value;
    }

    // Increase the number of columns to >= new_ncol
    void Grow(size_t new_ncol) {
        if (new_ncol <= num_columns_) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        if (new_ncol > column_capacity_) { // Need new segment(s)
            while (new_ncol > column_capacity_) {
                std::cout << "ConcurrentMatrix: Growing from "
                          << column_capacity_ << " to ";
                const auto segment_size = safe_left_shift(base_column_shift_ + row_shift_ + num_);
                data_[num_].reset(new std::atomic<T>[segment_size]);
                std::memset(data_[num_].get(), 0, s_ * segment_size);
                column_capacity_ += uint64_t{1} << (base_column_shift_ + num_);
                std::cout << column_capacity_ << " columns.\n";
                ++num_;
            }
        }
        if (new_ncol > num_columns_) num_columns_ = new_ncol;
std::cout << "ConcurrentMatrix: Grow to "
                  << num_columns_ << " columns.\n";
    }

    // Move all current data to a single segment
    void Consolidate() {
        if (num_ == 1) return;
        std::lock_guard<std::mutex> guard(mutex_);
        // Round to next power-of-two
        while (column_capacity_ != lowbit(column_capacity_))
            column_capacity_ += lowbit(column_capacity_);
        uint64_t segment_size = (uint64_t{1} << row_shift_) * column_capacity_;
        auto* new_data = new std::atomic<T>[segment_size];
        std::memset(new_data, 0, s_ * segment_size);
        size_t c_index = 0;
        for (int n = 0; n < num_; ++n) { // all but the last are full
            const auto C = uint64_t{1} << (base_column_shift_ + n);
            const auto seg_bytes = C * s_;
            for (size_t r = 0; r <= num_rows_; ++r) {
                std::memcpy(new_data + r * column_capacity_ + c_index,
                            data_[n].get() + r * C, seg_bytes);
            }
            c_index += C;
        }
        data_[0].reset(new_data);
        base_column_shift_ = bsr(column_capacity_);
        std::cout << "ConcurrentMatrix: Consolidated from "
                  << num_ << " segment to one with "
                  << column_capacity_<<"(2^"<<base_column_shift_<<") cols.\n";
        for (int i = 1; i < 64; ++i)
            data_[i].reset(nullptr);
        num_ = 1;
    }

    std::pair<std::atomic<T>*, std::size_t> raw_buffer() noexcept {
        if (num_ > 1) {
            Consolidate();
        }
        const std::size_t n_rows = num_rows_ + 1;
        const std::size_t n_cols = column_capacity_;
        return { data_[0].get(), n_rows * n_cols };
    }

private:

    inline void find_bucket(size_t c, int& bucket, size_t& bucket_c) const noexcept {
        if (c < (uint64_t{1} << base_column_shift_)) {
            bucket = 0;
            bucket_c = c;
        } else {
            bucket = bsr((c >> base_column_shift_) + 1);
            bucket_c = c - ((uint64_t{1} << bucket) - 1) *
                           (uint64_t{1} << base_column_shift_);
        }
    }

    std::atomic<T>& At(size_t r, size_t c) const noexcept {
        int bucket;
        size_t bucket_c;
        find_bucket(c, bucket, bucket_c);
        return data_[bucket][(r << (base_column_shift_ + bucket)) + bucket_c];
    }

    int num_; // number of segments currently allocated
    int row_shift_; // bits used for row index
    int base_column_shift_; // bits used for col index in the first segment
    size_t num_rows_, num_columns_; // actual number of rows and columns
    uint64_t column_capacity_; // total allocated columns
    std::array<Segment, 64>  data_{}; // ptrs to segments
    mutable std::mutex mutex_; // guard memory layout changes
    size_t s_;
};
