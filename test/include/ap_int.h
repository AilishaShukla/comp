// Minimal stand-in for Xilinx ap_int.h, for compiling HLS kernels on a laptop.
//
// Supports what the kernels use: ap_int<W> / ap_uint<W> of any width, bit
// slices x(hi, lo) / x.range(hi, lo) for read and write, construction from
// integers and slices, and conversion to a native integer for W <= 64.
// Arithmetic is done on the native integer after conversion.
//
// Bounds are checked on every slice: an out-of-range slice exits with a
// message instead of silently corrupting data. This is NOT the real header;
// HLS-specific errors still only show up in Vitis.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

template <int W, bool S> class ap_int_base;
template <int W, bool S> class ap_range_ref;

namespace ap_detail {

// Read up to 64 bits [lo, lo+n) from a word array.
inline uint64_t get_bits(const uint64_t *w, int lo, int n) {
    const int i = lo >> 6, o = lo & 63;
    uint64_t v = w[i] >> o;
    if (o && o + n > 64) v |= w[i + 1] << (64 - o);
    return n == 64 ? v : v & ((1ULL << n) - 1);
}

// Write n <= 64 bits of v into [lo, lo+n).
inline void set_bits(uint64_t *w, int lo, int n, uint64_t v) {
    const uint64_t m = n == 64 ? ~0ULL : (1ULL << n) - 1;
    v &= m;
    const int i = lo >> 6, o = lo & 63;
    w[i] = (w[i] & ~(m << o)) | (v << o);
    if (o && o + n > 64) {
        const int r = 64 - o;
        const uint64_t m2 = m >> r;
        w[i + 1] = (w[i + 1] & ~m2) | (v >> r);
    }
}

// Copy n bits from src[slo..] to dst[dlo..], 64 at a time.
inline void copy_bits(uint64_t *dst, int dlo, const uint64_t *src, int slo, int n) {
    for (int k = 0; k < n; k += 64) {
        const int c = n - k < 64 ? n - k : 64;
        set_bits(dst, dlo + k, c, get_bits(src, slo + k, c));
    }
}

}  // namespace ap_detail

template <int W, bool S> class ap_int_base {
    static_assert(W > 0, "width must be positive");

public:
    static const int NW = (W + 63) / 64;
    uint64_t w[NW];

    ap_int_base() { fill(0); }
    ap_int_base(long long v) { from_i64(v); }
    ap_int_base(unsigned long long v) { from_u64(v); }
    ap_int_base(long v) { from_i64(v); }
    ap_int_base(unsigned long v) { from_u64(v); }
    ap_int_base(int v) { from_i64(v); }
    ap_int_base(unsigned v) { from_u64(v); }

    template <int W2, bool S2> ap_int_base(const ap_int_base<W2, S2> &o) {
        fill(0);
        const int n = W < W2 ? W : W2;
        ap_detail::copy_bits(w, 0, o.w, 0, n);
        if (W > W2 && S2 && o.bit(W2 - 1)) sign_fill(W2);
        trim();
    }

    // A slice is unsigned: it is zero-extended, then reinterpreted at width W.
    template <int W2, bool S2> ap_int_base(const ap_range_ref<W2, S2> &r) {
        fill(0);
        const int n = W < r.width() ? W : r.width();
        ap_detail::copy_bits(w, 0, r.p->w, r.lo, n);
        trim();
    }

    bool bit(int i) const { return (w[i >> 6] >> (i & 63)) & 1ULL; }

    // Native value (W <= 64): sign-extended if signed, zero-extended if not.
    long long to_int64() const {
        uint64_t v = w[0];
        if (W < 64 && S && ((v >> (W - 1)) & 1ULL)) v |= ~((1ULL << W) - 1);
        return (long long)v;
    }
    operator long long() const {
        static_assert(W <= 64, "convert a slice of a wide ap_int, not the whole value");
        return to_int64();
    }

    ap_range_ref<W, S> operator()(int hi, int lo) const {
        return ap_range_ref<W, S>(const_cast<ap_int_base *>(this), hi, lo);
    }
    ap_range_ref<W, S> range(int hi, int lo) const { return (*this)(hi, lo); }

private:
    void fill(uint64_t v) {
        for (int i = 0; i < NW; i++) w[i] = v;
    }
    void from_i64(long long v) {
        fill(v < 0 ? ~0ULL : 0ULL);
        w[0] = (uint64_t)v;
        trim();
    }
    void from_u64(unsigned long long v) {
        fill(0);
        w[0] = v;
        trim();
    }
    void sign_fill(int from) {
        for (int i = from; i < W; i++) w[i >> 6] |= 1ULL << (i & 63);
    }
    void trim() {
        if (W % 64) w[NW - 1] &= (1ULL << (W % 64)) - 1;
    }
};

template <int W, bool S> class ap_range_ref {
public:
    ap_int_base<W, S> *p;
    int hi, lo;

    ap_range_ref(ap_int_base<W, S> *p_, int hi_, int lo_) : p(p_), hi(hi_), lo(lo_) {
        if (lo < 0 || hi >= W || lo > hi) {
            std::fprintf(stderr, "\n*** BIT SLICE OUT OF RANGE: x(%d, %d) on a %d-bit value (valid bits 0..%d) ***\n",
                         hi, lo, W, W - 1);
            std::exit(2);
        }
    }
    int width() const { return hi - lo + 1; }

    unsigned long long to_uint64() const {
        return ap_detail::get_bits(p->w, lo, width() < 64 ? width() : 64);
    }
    operator unsigned long long() const { return to_uint64(); }

    ap_range_ref &operator=(long long v) {
        const int n = width();
        for (int k = 0; k < n; k += 64) {
            const int c = n - k < 64 ? n - k : 64;
            ap_detail::set_bits(p->w, lo + k, c, k == 0 ? (uint64_t)v : (v < 0 ? ~0ULL : 0ULL));
        }
        return *this;
    }
    ap_range_ref &operator=(int v) { return *this = (long long)v; }
    ap_range_ref &operator=(unsigned long long v) { return *this = (long long)v; }

    template <int W2, bool S2> ap_range_ref &operator=(const ap_int_base<W2, S2> &o) {
        const int n = width() < W2 ? width() : W2;
        ap_detail::copy_bits(p->w, lo, o.w, 0, n);
        if (width() > W2) {
            const long long ext = (S2 && o.bit(W2 - 1)) ? -1 : 0;
            ap_range_ref(p, hi, lo + W2) = ext;
        }
        return *this;
    }
    ap_range_ref &operator=(const ap_range_ref &o) {
        const ap_int_base<512, false> t(o);
        return *this = t;
    }
};

template <int W> class ap_int : public ap_int_base<W, true> {
public:
    using ap_int_base<W, true>::ap_int_base;
    ap_int() = default;
};

template <int W> class ap_uint : public ap_int_base<W, false> {
public:
    using ap_int_base<W, false>::ap_int_base;
    ap_uint() = default;
};
