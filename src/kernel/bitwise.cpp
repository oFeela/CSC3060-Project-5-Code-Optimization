#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// TODO: Optimize the bitwise function
// TODO: TRY SIMD WITH HIGHER COUNT
// TODO: TRY MULTITHREADING SINCE DATA IS INDEPENDENT, NO RACING CONDS
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    // Implement your version...
    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    std::size_t i = 0;

    // explicit call for prefetch: __builtin_prefetch(addr, rw, locality)
    // loops prefetch alr done by the flag in CMakeLists.txt

    // 16 at a time
    // constexpr __uint128_t kMaskLoP128 = (__uint128_t)0x5A5A5A5A5A5A5A5A << 64 | 0x5A5A5A5A5A5A5A5A;
    // constexpr __uint128_t kMaskHi128 = (__uint128_t)0xC3C3C3C3C3C3C3C3 << 64 | 0xC3C3C3C3C3C3C3C3;

    // for (; i + 15 < n; i += 16) {
    //     __uint128_t ap, bp;
    //     memcpy(&ap, &a[i], 16);
    //     memcpy(&bp, &b[i], 16);
        
    //     __uint128_t shared = ap & bp;
    //     __uint128_t either = ap | bp;
    //     __uint128_t diff = ap ^ bp;
    //     __uint128_t mixed0 = (diff & kMaskLoP128) | (~shared & ~kMaskLoP128);
    //     __uint128_t mixed1 = ((either ^ kMaskHi128) & (shared | ~kMaskHi128)) ^ diff;
        
    //     __uint128_t res = mixed0 ^ mixed1;
        
    //     memcpy(&result[i], &res, 16);
    // }

    // // 8 at a time
    // constexpr std::uint64_t kMaskLo64 = 0x5A5A5A5A5A5A5A5Au;
    // constexpr std::uint64_t kMaskHi64 = 0xC3C3C3C3C3C3C3C3u;
    // for (; i + 7 < n; i += 8) {
    //     std::uint64_t ap, bp;
    //     // get four elements starting from curr
    //     memcpy(&ap, &a[i], 8);
    //     memcpy(&bp, &b[i], 8);

    //     std::uint64_t shared = ap & bp;
    //     std::uint64_t either = ap | bp;
    //     std::uint64_t diff = ap ^ bp;
    //     std::uint64_t mixed0 = (diff & kMaskLo64) | (~shared & ~kMaskLo64);
    //     std::uint64_t mixed1 = ((either ^ kMaskHi64) & (shared | ~kMaskHi64)) ^ diff;
        
    //     std::uint64_t res = mixed0 ^ mixed1;

    //     memcpy(&result[i], &res, 8);
    // }

    // // 4 at a time
    // constexpr std::uint32_t kMaskLo32 = 0x5A5A5A5Au;
    // constexpr std::uint32_t kMaskHi32 = 0xC3C3C3C3u;
    // for (; i + 3 < n; i += 4) {
    //     std::uint32_t ap, bp;
    //     // get four elements starting from curr
    //     memcpy(&ap, &a[i], 4);
    //     memcpy(&bp, &b[i], 4);

    //     std::uint32_t shared = ap & bp;
    //     std::uint32_t either = ap | bp;
    //     std::uint32_t diff = ap ^ bp;
    //     std::uint32_t mixed0 = (diff & kMaskLo32) | (~shared & ~kMaskLo32);
    //     std::uint32_t mixed1 = ((either ^ kMaskHi32) & (shared | ~kMaskHi32)) ^ diff;
        
    //     std::uint32_t res = mixed0 ^ mixed1;

    //     memcpy(&result[i], &res, 4);
    // }

    // // 2 at a time
    // constexpr std::uint16_t kMaskLo16 = 0x5A5Au;
    // constexpr std::uint16_t kMaskHi16 = 0xC3C3u;
    // for (; i + 1 < n; i += 2) {
    //     std::uint16_t ap, bp;
    //     // get four elements starting from curr
    //     memcpy(&ap, &a[i], 2);
    //     memcpy(&bp, &b[i], 2);

    //     std::uint16_t shared = ap & bp;
    //     std::uint16_t either = ap | bp;
    //     std::uint16_t diff = ap ^ bp;
    //     std::uint16_t mixed0 = (diff & kMaskLo16) | (~shared & ~kMaskLo16);
    //     std::uint16_t mixed1 = ((either ^ kMaskHi16) & (shared | ~kMaskHi16)) ^ diff;
        
    //     std::uint16_t res = mixed0 ^ mixed1;

    //     memcpy(&result[i], &res, 2);
    // }

    // leftovers or simply just naive one, preferred with flags on! (-O3)
    constexpr std::uint8_t MaskLo = 0x5Au;
    constexpr std::uint8_t MaskHi = 0xC3u;
    for (; i < n; i++) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & MaskLo) | (~shared & ~MaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ MaskHi) & (shared | ~MaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    // Call your verion here
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}
