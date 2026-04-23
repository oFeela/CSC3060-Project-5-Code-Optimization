#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

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
        }
    }
}

// TODO need to fix the accuracy
void stu_matmul(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int n) {
    #if 1
    std::fill(C.begin(), C.end(), 0.0f);

    for (int r = 0; r < n; r++) {
        for (int k = 0; k < n; k++) {
            for (int c = 0; c + 7 < n; c += 8) {
                C[r * n + c] += A[r * n + k] * B[k * n + c + 0];
                C[r * n + c + 1] += A[r * n + k] * B[k * n + c + 1];
                C[r * n + c + 2] += A[r * n + k] * B[k * n + c + 2];
                C[r * n + c + 3] += A[r * n + k] * B[k * n + c + 3];
                C[r * n + c + 4] += A[r * n + k] * B[k * n + c + 4];
                C[r * n + c + 5] += A[r * n + k] * B[k * n + c + 5];
                C[r * n + c + 6] += A[r * n + k] * B[k * n + c + 6];
                C[r * n + c + 7] += A[r * n + k] * B[k * n + c + 7];
                // C[r * n + c + 8] += A[r * n + k] * B[k * n + c + 8];
                // C[r * n + c + 9] += A[r * n + k] * B[k * n + c + 9];
                // C[r * n + c + 10] += A[r * n + k] * B[k * n + c + 10];
                // C[r * n + c + 11] += A[r * n + k] * B[k * n + c + 11];
                // C[r * n + c + 12] += A[r * n + k] * B[k * n + c + 12];
                // C[r * n + c + 13] += A[r * n + k] * B[k * n + c + 13];
                // C[r * n + c + 14] += A[r * n + k] * B[k * n + c + 14];
                // C[r * n + c + 15] += A[r * n + k] * B[k * n + c + 15];
            }
        }
    }
    #endif
     
    // precision loss
    #if 0
    std::vector<float> BT(n * n);
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            BT[r * n + c] = B[c * n + r];
        }
    }

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            float sum = 0.0f;
            for (int k = 0; k + 7 < n; k += 8) {
                sum += A[r * n + k] * BT[c * n + k];
                sum += A[r * n + k + 1] * BT[c * n + k + 1];
                sum += A[r * n + k + 2] * BT[c * n + k + 2];
                sum += A[r * n + k + 3] * BT[c * n + k + 3];
                sum += A[r * n + k + 4] * BT[c * n + k + 4];
                sum += A[r * n + k + 5] * BT[c * n + k + 5];
                sum += A[r * n + k + 6] * BT[c * n + k + 6];
                sum += A[r * n + k + 7] * BT[c * n + k + 7];
            }
            C[r * n + c] = sum;
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
