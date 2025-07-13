#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <algorithm>
#include <iostream>

// A dynamic column-major matrix
template<class T>
class Matrix {
public:
    Matrix(int R = 1, int C = 1)
        : R(R), C(C), data(R * C), Rcap(R), Ccap(C) {}

    /* ------------ dimension & capacity getters --------------------- */
    int  GetR()    const { return R;    }
    int  GetC()    const { return C;    }
    int  GetRcap() const { return Rcap; }
    int  GetCcap() const { return Ccap; }
    size_t GetSize() const { return static_cast<size_t>(C) * Rcap; }

    /* ------------ resize helpers (fit = exact /  fill = zero-init) -- */
    void SetR(int new_R, bool fit=false, bool fill=false)
    { Resize(new_R, C, fit, fill); }

    void SetC(int new_C, bool fit=false, bool fill=false)
    { Resize(R, new_C, fit, fill); }

    void Resize(int new_R, int new_C, bool fit = false, bool fill = false) {
        if (new_R <= Rcap && new_C <= Ccap) {   // no reallocation needed
            if (fill) {
                // set elements outside the new R, C to zero
                for (int c = 0; c < C; ++c)
                    std::fill(data.begin() + c * Rcap + new_R,
                              data.begin() + (c + 1) * Rcap, T{});
                for (int c = C; c < Ccap; ++c)
                    std::fill(data.begin() + c * Rcap,
                              data.begin() + (c + 1) * Rcap, T{});
            }
            R = new_R; C = new_C;
            return;
        }

        const int old_Rcap = Rcap, old_Ccap = Ccap;

        /* --- grow capacities geometrically unless “fit” asked for ---- */
        if (fit) { Rcap = new_R; Ccap = new_C; }
        else {
            while (Rcap < new_R) Rcap = Rcap * 2 + 1;
            while (Ccap < new_C) Ccap = Ccap * 2 + 1;
        }

        std::vector<T> old = std::move(data);
        data.resize(static_cast<size_t>(Rcap) * Ccap);
        /* ------------- copy old -> new, column by column ------------- */
        for (int c = 0; c < C; ++c) {
            std::copy(old.begin() + c * old_Rcap,
                      old.begin() + c * old_Rcap + R, data.begin() + c * Rcap);

            if (fill)            // zero-pad new rows in existing cols
                std::fill(data.begin() + c * Rcap + R,
                          data.begin() + (c + 1) * Rcap, T{});
        }
        if (fill) {             // zero-pad brand-new columns
            for (int c = C; c < Ccap; ++c)
                std::fill(data.begin() + c * Rcap,
                          data.begin() + (c + 1) * Rcap, T{});
        }

        R = new_R;  C = new_C;
    }

    /* ------------ element access ----------------------------------- */
    T& operator()(int r, int c)               { return data[c * Rcap + r]; }
    const T& operator()(int r, int c) const   { return data[c * Rcap + r]; }

    /* column-contiguous access */
    T* ColPtr(int c)             { return &data[c * Rcap]; }
    const T* ColPtr(int c) const { return &data[c * Rcap]; }
    T*       Data()       { return data.data(); }
    const T* Data() const { return data.data(); }

    /* set a column */
    void SetColumn(int c, const T* src) {
        if (c >= C)              // auto-expand if necessary
            Resize(R, c + 1);
        std::memcpy(&data[c * Rcap], src, sizeof(T) * R);
    }
    template<class InputIt>
    void SetColumn(int c, InputIt first, InputIt last, bool fill_rest = false) {
        if (c >= C)
            Resize(R, c + 1);
        T* dest = &data[c * Rcap];
        size_t i = 0;
        for (; first != last && i < static_cast<size_t>(R); ++first, ++i)
            dest[i] = static_cast<T>(*first);

        if (fill_rest)               // optionally clear unused rows
            std::fill(dest + i, dest + R, T{});
    }

    void SetZero()
    { std::memset(data.data(), 0, sizeof(T) * data.size()); }



private:
    int R, Rcap;   // logical rows / capacity
    int C, Ccap;   // logical cols / capacity
    std::vector<T> data;   // column-major: element(r,c) == data[c*Rcap + r]
};
