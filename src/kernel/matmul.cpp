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
    // TODO: Implement your version, and call it in stu_matmul_wrapper
    #if 1 // TODO mine
    std::vector<double> C_double(C.size(), 0.0); // Use double for accumulation
    // std::fill(C.begin(), C.end(), 0.0f); // zeros out first, so we can accumulate later
    constexpr int block_size = 64; // TODO (change later) matrix partition block size

    for (int start_i = 0; start_i < n; start_i += block_size) {
        for (int start_j = 0; start_j < n; start_j += block_size) {
            // block multiplication
            for (int start_k = 0; start_k < n; start_k += block_size) {
                for (int i = start_i; i < std::min(n, start_i + block_size); i++) {
                    for (int j = start_j; j < std::min(n, start_j + block_size); j++) {
                        double sum = C_double[i * n + j];
                        for (int k = start_k; k < std::min(n, start_k + block_size); k++) {
                            sum += (double) A[i * n + k] * B[k * n + j];
                        }
                        C_double[i * n + j] = sum;
                    }
                }
            }

            // // output
            // for (int i = start_i; i < std::min(n, start_i + block_size); i++)
            //     for (int j = start_j; j < std::min(n, start_j + block_size); j++)
            //         std::cout << "opti [" << i << "][" << j << "] = " << C[i * n + j] << std::endl; // TODO remove
        }
    }

    // convert back to float from C_double
    for (size_t i = 0; i < C.size(); i++) {
        C[i] = static_cast<float>(C_double[i]);
    }
    #endif

    #if 0 // TODO partner's
    std::fill(C.begin(), C.end(), 0.0f);

    constexpr int block_size = 64;

    const float *const a_ptr = A.data();
    const float *const b_ptr = B.data();
    float *const c_ptr = C.data();

    for (int start_i = 0; start_i < n; start_i += block_size) {
        const int end_i = std::min(n, start_i + block_size);
        for (int start_k = 0; start_k < n; start_k += block_size) {
            const int end_k = std::min(n, start_k + block_size);
            for (int start_j = 0; start_j < n; start_j += block_size) {
                const int end_j = std::min(n, start_j + block_size);
                const int j_block = end_j - start_j;

                for (int i = start_i; i < end_i; ++i) {
                    const size_t row_offset =
                        static_cast<size_t>(i) * static_cast<size_t>(n);
                    const float *const a_row = a_ptr + row_offset;
                    float *const c_row = c_ptr + row_offset + start_j;

                    for (int k = start_k; k < end_k; ++k) {
                        const float a_val = a_row[k];
                        const float *const b_row =
                            b_ptr +
                            static_cast<size_t>(k) * static_cast<size_t>(n) +
                            start_j;

                        int j = 0;
                        for (; j + 7 < j_block; j += 8) {
                            c_row[j + 0] += a_val * b_row[j + 0];
                            c_row[j + 1] += a_val * b_row[j + 1];
                            c_row[j + 2] += a_val * b_row[j + 2];
                            c_row[j + 3] += a_val * b_row[j + 3];
                            c_row[j + 4] += a_val * b_row[j + 4];
                            c_row[j + 5] += a_val * b_row[j + 5];
                            c_row[j + 6] += a_val * b_row[j + 6];
                            c_row[j + 7] += a_val * b_row[j + 7];
                        }
                        for (; j < j_block; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
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
            return false;
        }
    }

    debug_log("\tDEBUG: matmul_check passed. max_rel={} at index {}\n",
              max_rel,
              worst_idx);
    return true;
}
