#include "relu.h"
#include <algorithm>
#include <cstdint>
#include <random>
#include <bit>
#include <thread>

void initialize_relu(relu_args *args, const size_t size,
                     const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr float mean = 0.0f;
    constexpr float stddev = 1.0f;

    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);

    args->data.resize(size);

    for (auto &value : args->data) {
        value = dist(gen);
    }
}

void naive_relu(std::span<float> data) {
    for (auto &&value : data) {
        if (value < 0.0f) {
            value = 0.0f;
        }
    }
}

void stu_relu(std::span<float> data) {
    // TODO: Implement your version, and call it in stu_relu_wrapper
    // TODO: try MULTITHREADING ALSO, slower :( --> probably due to the intialization overhead
    #if 0
    size_t n = data.size();
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    size_t chunk_size = n / num_threads;

    for (unsigned int t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? n : start + chunk_size;
        
        threads.emplace_back([&data, start, end]() {
            for (size_t i = start; i < end; i++) {
                // data[i] = std::max(0.0f, data[i]);
                int32_t b = std::bit_cast<int32_t>(data[i]);
                b &= ~(b >> 31);
                data[i] = std::bit_cast<float>(b);
            }
        });
    }

    for (auto& t : threads) t.join();
    #endif

    #if 1
    size_t n = data.size();
    for (size_t i = 0; i < n; i++) {
        // x & ~(x >> 31) = x
        // 0b1110 < 0, for x < 0, 0b1110 >> 3 = 0b1111, negate it and becomes 0b0000
        // x & ~(x >> 31) = 0
        // val &= ~(val >> 31);

        // get address where val is
        // typecast as a pointer to int32_t
        // dereference it --> get int32_t

        /*
        This one also okay with flags on
        */
        // int32_t& bit_val = *(int32_t*)&val; 
        // bit_val &= ~(bit_val >> 31);

        /*
        No ptr arith?
        */
        // int32_t b = std::bit_cast<int32_t>(val);
        // b &= ~(b >> 31);
        // val = std::bit_cast<float>(b);

        /*
        Fastest for now! (below BASELINE)
        TODO: OPTIMIZE MORE, SIMD maybe?
        */
        data[i] = std::max(0.0f, data[i]);
    }
    #endif
}

void naive_relu_wrapper(void *ctx) {
    auto &args = *static_cast<relu_args *>(ctx);
    naive_relu(args.data);
}

void stu_relu_wrapper(void *ctx) {
    auto &args = *static_cast<relu_args *>(ctx);
    stu_relu(args.data);
}

bool relu_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<relu_args *>(stu_ctx);
    auto &ref_args = *static_cast<relu_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;

    if (stu_args.data.size() != ref_args.data.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.data.size(),
                  ref_args.data.size());
        return false;
    }

    double max_rel = 0.0;
    size_t worst_i = 0;
    const double atol = 1e-6;

    for (size_t i = 0; i < ref_args.data.size(); ++i) {
        const double r = static_cast<double>(ref_args.data[i]);
        const double s = static_cast<double>(stu_args.data[i]);
        const double err = std::abs(s - r);
        const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;

        if (rel > max_rel) {
            max_rel = rel;
            worst_i = i;
        }

        if (err > (atol + eps * std::abs(r))) {
            debug_log("\tDEBUG: fail at {}: ref={} stu={} err={} rel={} thr={}\n",
                      i,
                      ref_args.data[i],
                      stu_args.data[i],
                      err,
                      rel,
                      (atol + eps * std::abs(r)));
            return false;
        }
    }

    debug_log("\tDEBUG: relu_check passed. max_rel={} at i={}\n",
              max_rel,
              worst_i);
    return true;
}
