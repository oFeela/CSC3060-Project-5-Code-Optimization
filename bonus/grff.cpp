#include "grff.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <thread>

void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    args->a_features.resize(size);
    args->b_features.resize(size);
    args->c_features.resize(size);
    args->f_output.resize(size);

    for (size_t i = 0; i < size; ++i) {
        args->a_features[i] = dist(gen);
        args->b_features[i] = dist(gen);
        args->c_features[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Naive Implementation (A Simplified Gated Residual Feature Fusion (GRFF))
// -------------------------------------------------------------------------
void naive_grff(grff_args& args) {
    size_t n = args.a_features.size();
    
    // Intermediate buffers
    std::vector<float> G(n), A_prime(n), Smooth_A(n), B_prime(n), C_prime(n), H(n), E(n);

    // Stage 1: Gate
    for (size_t i = 0; i < n; ++i) 
        G[i] = 0.5f * ((args.a_features[i] * args.b_features[i]) / (1.0f + std::abs(args.a_features[i] * args.b_features[i])) + 1.0f);

    // Stage 2: Update A (Residual)
    for (size_t i = 0; i < n; ++i) 
        A_prime[i] = args.a_features[i] + G[i];

    // Stage 3: Global Feature Scaling
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_a += A_prime[i];
    }
    float avg_a = sum_a / static_cast<float>(n);

    // Stage 4: Update A (Smooth)
    Smooth_A[0] = A_prime[0];
    for (size_t i = 1; i < n; ++i) {
        Smooth_A[i] = (A_prime[i] + A_prime[i-1]) * 0.5f; 
    }

    // Stage 5: Update B (Suppression)
    for (size_t i = 0; i < n; ++i) 
        B_prime[i] = args.b_features[i] * (1.0f - G[i]) * avg_a;

    // Stage 6: Context Integration 
    for (size_t i = 0; i < n; ++i) 
        C_prime[i] = args.c_features[i] + (Smooth_A[i] / (1.0f + std::abs(Smooth_A[i])));

    // Stage 7: Hidden Interaction
    for (size_t i = 0; i < n; ++i) 
        H[i] = Smooth_A[i] * C_prime[i];

    // Stage 8: Normalization
    for (size_t i = 0; i < n; ++i) 
        E[i] = (H[i] + B_prime[i]) / (1.0f + std::abs(Smooth_A[i]));

    // Stage 9: Final Output (ReLU)
    for (size_t i = 0; i < n; ++i) {
        float result = C_prime[i] - E[i];
        args.f_output[i] = std::max(result, 0.0f);
    }
}

// -------------------------------------------------------------------------
// TODO: Student Implementation
// -------------------------------------------------------------------------
void stu_grff(grff_args& args) {
    #if 0 // better than naive, but 0.827x of the BASELINE
    float sum_a = 0.0f; // for Stage 3: Global Feature Scaling
    bool first = true; // my addition, for Stage 4: Update A (Smooth)
    for (size_t i = 0; i < n; ++i) {
        // Stage 1: Gate
        G[i] = 0.5f * ((args.a_features[i] * args.b_features[i]) / (1.0f + std::abs(args.a_features[i] * args.b_features[i])) + 1.0f); // updates G[i]
        
        // Stage 2: Update A (Residual)
        A_prime[i] = args.a_features[i] + G[i]; // needs updated G[i], updates A_prime[i]

        // Stage 3: Global Feature Scaling
        sum_a += A_prime[i]; // needs updated A_prime[i]

        // Stage 4: Update A (Smooth)
        if (first) { // TODO hopefully predicts well, only true once
            Smooth_A[0] = A_prime[0];
            first = false;
        }
        else Smooth_A[i] = (A_prime[i] + A_prime[i-1]) * 0.5f;
        
    }
    float avg_a = sum_a / static_cast<float>(n); // for Stage 3: Global Feature Scaling

    // this goes after avg_a is computed outside above loop
    for (size_t i = 0; i < n; ++i) {
        // Stage 5: Update B (Suppression)
        B_prime[i] = args.b_features[i] * (1.0f - G[i]) * avg_a; // need avg_a
        
        // Stage 6: Context Integration 
        C_prime[i] = args.c_features[i] + (Smooth_A[i] / (1.0f + std::abs(Smooth_A[i])));

        // Stage 7: Hidden Interaction
        H[i] = Smooth_A[i] * C_prime[i];

        // Stage 8: Normalization
        E[i] = (H[i] + B_prime[i]) / (1.0f + std::abs(Smooth_A[i]));

        // Stage 9: Final Output (ReLU)
        float result = C_prime[i] - E[i]; // needs updated C_prime[i], E[i]
        args.f_output[i] = std::max(result, 0.0f);
    }
    #endif

    // NO VECTOR USED
    // this is faster on my local machine, slower on server than below
    #if 0
    size_t n = args.a_features.size();
    double sum_a = 0.0f;

    // the rest
    for (size_t i = 1; i < n; i++) {
        float p = args.a_features[i] * args.b_features[i];
        float G = 0.5f * (p / (1.0f + std::abs(p)) + 1.0f);
        float A = args.a_features[i] + G; // A_prime

        sum_a += A;
    }
    float avg_a = sum_a / static_cast<float>(n);
    // above should be ok now! (i think, only 2 vector yayy)

    for (size_t i = 0; i < n; i++) {
        float p = args.a_features[i] * args.b_features[i];
        float G = 0.5f * (p / (1.0f + std::abs(p)) + 1.0f);
        float A = args.a_features[i] + G;
        float SA;
        if (i == 0) SA = A;
        else {
            float pp = args.a_features[i - 1] * args.b_features[i - 1];
            float Gp = 0.5f * (pp / (1.0f + std::abs(pp)) + 1.0f);
            float Ap = args.a_features[i - 1] + Gp;
            SA = (A + Ap) * 0.5f;
        } 

        float B = args.b_features[i] * (1.0f - G) * avg_a;
        float C = args.c_features[i] + (SA / (1.0f + std::abs(SA)));
        float H = SA * C;
        float E = (H + B) / (1.0f + std::abs(SA));
        float res = C - E;
        args.f_output[i] = std::max(res, 0.0f);
    }
    #endif

    // 1 VECTOR
    // FASTERST ON SERVER
    // possibly because the server's cpu can't do floating point arith fast enough
    // so storing the result in vector A is better
    #if 1
    size_t n = args.a_features.size();
    std::vector<float> A(n);
    float sum_a = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float p = args.a_features[i] * args.b_features[i];
        float G = 0.5f * 
        ((p) / 
        (1.0f + std::abs(p)) + 1.0f);

        A[i] = args.a_features[i] + G;
        sum_a += A[i];
    }
    float avg_a = sum_a / static_cast<float>(n);

    // first elem only
    float p = args.a_features[0] * args.b_features[0];
    float G = 0.5f * 
    ((p) / 
    (1.0f + std::abs(p)) + 1.0f);
    float SA = A[0];
    float B = args.b_features[0] * (1.0f - G) * avg_a;
    float C = args.c_features[0] + (SA / (1.0f + std::abs(SA)));
    float H = SA * C;
    float E = (H + B) / (1.0f + std::abs(SA));
    float res = C - E;
    args.f_output[0] = std::max(res, 0.0f);

    for (size_t i = 1; i < n; i++) {
        float p = args.a_features[i] * args.b_features[i];
        float G = 0.5f * 
        ((p) / 
        (1.0f + std::abs(p)) + 1.0f);

        float SA = (A[i] + A[i - 1]) * 0.5f;
        float B = args.b_features[i] * (1.0f - G) * avg_a;
        float C = args.c_features[i] + (SA / (1.0f + std::abs(SA)));
        float H = SA * C;
        float E = (H + B) / (1.0f + std::abs(SA));
        float res = C - E;
        args.f_output[i] = std::max(res, 0.0f);
    }
    #endif

    #if 0
    size_t n = args.a_features.size();
    std::vector<float> G(n), A(n);
    float sum_a = 0.0f;
    float prev_A_prime = 0.0f; // for smooth_A purpose

    // only for the first element
    float p = args.a_features[0] * args.b_features[0];

    G[0] = 0.5f * (p / (1.0f + std::abs(p)) + 1.0f);
    A[0] = args.a_features[0] + G[0]; // A_prime

    sum_a += A[0];

    prev_A_prime = A[0];

    // the rest
    for (size_t i = 1; i < n; i++) {
        p = args.a_features[i] * args.b_features[i];

        G[i] = 0.5f * (p / (1.0f + std::abs(p)) + 1.0f);
        A[i] = args.a_features[i] + G[i]; // A_prime

        sum_a += A[i];

        float temp = A[i];
        // smooth_A
        A[i] = (A[i] + prev_A_prime) * 0.5f;
        prev_A_prime = temp;
    }
    float avg_a = sum_a / static_cast<float>(n);
    // above should be ok now! (i think, only 2 vector yayy)

    #if 0
    for (size_t i = 0; i < n; i++) {
        float B = args.b_features[i] * (1.0f - G[i]) * avg_a;
        float C = args.c_features[i] + (A[i] / (1.0f + std::abs(A[i])));
        float H = A[i] * C;
        float E = (H + B) / (1.0f + std::abs(A[i]));
        float res = C - E;
        args.f_output[i] = std::max(res, 0.0f);
    }
    #endif

    // mutlithread GOOD also wow
    // key takeaway: use multithreading when one iteration of a loop DOES LOTS OF computation
    // obviously u neeed to ensure independence
    #if 1
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk_size = n / num_threads;

    for (unsigned int t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? n : start + chunk_size;
        
        threads.emplace_back([&, start, end]() {
            for (size_t i = start; i < end; i++) {
                float B = args.b_features[i] * (1.0f - G[i]) * avg_a;
                float C = args.c_features[i] + (A[i] / (1.0f + std::abs(A[i])));
                float H = A[i] * C;
                float E = (H + B) / (1.0f + std::abs(A[i]));
                float res = C - E;
                args.f_output[i] = std::max(res, 0.0f);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
    #endif
    #endif

    // 3 vectors ionstead of 2
    // theres preicsion loss here :(
    #if 0
    size_t n = args.a_features.size();
    std::vector<float> G(n), A(n), SA(n);
    float sum_a = 0.0f;

    // only for the first element
    float p = args.a_features[0] * args.b_features[0];

    G[0] = 0.5f * (p / (1.0f + std::abs(p)) + 1.0f);
    A[0] = args.a_features[0] + G[0]; // A_prime

    sum_a += A[0];
    SA[0] = A[0];

    // the rest
    for (size_t i = 1; i < n; i++) {
        p = args.a_features[i] * args.b_features[i];

        G[i] = 0.5f * (p / (1.0f + std::abs(p)) + 1.0f);
        A[i] = args.a_features[i] + G[i]; // A_prime

        sum_a += A[i];

        // smooth_A
        SA[i] = (A[i] + A[i - 1]) * 0.5f;
    }
    float avg_a = sum_a / static_cast<float>(n);
    // above should be ok now! (i think, only 2 vector yayy)

    for (size_t i = 0; i < n; i++) {
        float B = args.b_features[i] * (1.0f - G[i]) * avg_a;
        float C = args.c_features[i] + (SA[i] / (1.0f + std::abs(SA[i])));
        float H = SA[i] * C;
        float E = (H + B) / (1.0f + std::abs(SA[i]));
        float res = C - E;
        args.f_output[i] = std::max(res, 0.0f);
    }
    #endif
}

// -------------------------------------------------------------------------
// Wrappers and Checker
// -------------------------------------------------------------------------
void naive_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    naive_grff(args);
}

void stu_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    stu_grff(args);
}

bool grff_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<grff_args *>(stu_ctx);
    auto &ref_args = *static_cast<grff_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;
    const double atol = 1e-6;

    if (stu_args.f_output.size() != ref_args.f_output.size()) return false;

    for (size_t i = 0; i < ref_args.f_output.size(); ++i) {
        double r = static_cast<double>(ref_args.f_output[i]);
        double s = static_cast<double>(stu_args.f_output[i]);
        double err = std::abs(s - r);

        if (err > (atol + eps * std::abs(r))) {
            // debug_log("DEBUG: GRFF fail at %zu: ref=%f stu=%f\n", i, r, s);
            debug_log("DEBUG: GRFF fail at {}: ref={} stu={}\n", i, r, s);
            return false;
        }
    }
    return true;
}
