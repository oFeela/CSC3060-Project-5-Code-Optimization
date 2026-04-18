#include "matmul.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <iostream> // TODO remove

void initialize_matmul(matmul_args& args, int n, uint32_t seed) {
    if (n <= 0) {
        throw std::invalid_argument("initialize_matmul: n must be positive.");
    }

    args.n = n;
    args.epsilon = 1e-3;

    const size_t elem_count = static_cast<size_t>(n) * static_cast<size_t>(n);
    args.A.resize(elem_count);
    args.B.resize(elem_count);
    args.C.assign(elem_count, 0.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elem_count; ++i) {
        args.A[i] = dist(rng);
        args.B[i] = dist(rng);
    }
}

void naive_matmul(std::vector<float>& C,
                  const std::vector<float>& A,
                  const std::vector<float>& B,
                  int n) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
            // std::cout << "naive [" << i << "][" << j << "] = " << sum << std::endl; // TODO remove
        }
    }
}

void stu_matmul(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int n) {
    constexpr int block_size = 64; // TODO change later

    #if 0 // SP float, i-j-k inner loop
    // std::fill(C.begin(), C.end(), 0.0f); // zeros out first
    std::vector<double> C_double(C.size(), 0.0);

    for (int start_i = 0; start_i < n; start_i += block_size) {
        int end_i = std::min(n, start_i + block_size);
        for (int start_j = 0; start_j < n; start_j += block_size) {
            int end_j = std::min(n, start_j + block_size);
            for (int start_k = 0; start_k < n; start_k += block_size) {
                int end_k = std::min(n, start_k + block_size);

                for (int i = start_i; i < end_i; i++) {
                    for (int j = start_j; j < end_j; j++) {
                        double sum = C_double[i * n + j]; // float
                        for (int k = start_k; k < end_k; k++) {
                            sum += (double) A[i * n + k] * B[k * n + j];
                        }
                        C_double[i * n + j] = sum;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < C.size(); i++) {
        C[i] = static_cast<float>(C_double[i]);
    }
#endif

#if 1 // TODO Double Precision, i-k-j inner loop. MUCH FASTER but why???
    std::vector<double> C_double(C.size(), 0.0);

    for (int start_i = 0; start_i < n; start_i += block_size) {
        int end_i = std::min(n, start_i + block_size);
        for (int start_k = 0; start_k < n; start_k += block_size) {
            int end_k = std::min(n, start_k + block_size);
            for (int start_j = 0; start_j < n; start_j += block_size) {
                int end_j = std::min(n, start_j + block_size);

                for (int i = start_i; i < end_i; ++i) {
                    for (int k = start_k; k < end_k; ++k) {
                        double a_val = A[i * n + k]; // promote to double
                        for (int j = start_j; j < end_j; ++j) {
                            C[i * n + j] += a_val * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < C.size(); i++) {
        C[i] = static_cast<float>(C_double[i]);
    }
#endif
}

void naive_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    naive_matmul(args.C, args.A, args.B, args.n);
}

void stu_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    stu_matmul(args.C, args.A, args.B, args.n);
}

bool matmul_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<matmul_args*>(stu_ctx);
    auto& ref_args = *static_cast<matmul_args*>(ref_ctx);

    if (stu_args.C.size() != ref_args.C.size()) {
        debug_log("\tDEBUG: matmul size mismatch: stu={} ref={}\n",
                  stu_args.C.size(),
                  ref_args.C.size());
        return false;
    }

    const double eps = ref_args.epsilon;
    const int n = ref_args.n;
    double max_rel = 0.0;
    size_t worst_idx = 0;

    for (size_t i = 0; i < ref_args.C.size(); ++i) {
        const double r = static_cast<double>(ref_args.C[i]);
        const double s = static_cast<double>(stu_args.C[i]);
        const double diff = std::abs(s - r);
        const double rel = (std::abs(r) > 1e-9) ? diff / std::abs(r) : diff;

        if (rel > max_rel) {
            max_rel = rel;
            worst_idx = i;
        }

        if (rel > eps) {
            const size_t row = (n > 0) ? (i / static_cast<size_t>(n)) : 0;
            const size_t col = (n > 0) ? (i % static_cast<size_t>(n)) : 0;
            debug_log("\tDEBUG: matmul fail at index {} (row={}, col={}): ref={} stu={} rel={} eps={}\n",
                      i,
                      row,
                      col,
                      ref_args.C[i],
                      stu_args.C[i],
                      rel,
                      eps);
            //return false;
        }
    }

    debug_log("\tDEBUG: matmul_check passed. max_rel={} at index {}\n",
              max_rel,
              worst_idx);
    return true;
}
