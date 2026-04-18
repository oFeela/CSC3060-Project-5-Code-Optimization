#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "relu.h"
#include "bitwise.h"
#include "matmul.h"


int main() {
    std::uint32_t seed = 12345u;
    constexpr size_t relu_size = 1024000;
    relu_args relu_args_naive;
    initialize_relu(&relu_args_naive, relu_size, seed);
    relu_args relu_args_student = relu_args_naive;
    std::println("\tReLU: vector length={}", relu_size);

    constexpr size_t bitwise_size = 1024000;
    bitwise_args bitwise_args_naive;
    initialize_bitwise(&bitwise_args_naive, bitwise_size, seed);
    bitwise_args bitwise_args_student = bitwise_args_naive;
    std::println("\tBitwise: vector length={}", bitwise_size);

    constexpr size_t matmul_size = 512; // TODO change to 512
    matmul_args matmul_args_naive;
    initialize_matmul(matmul_args_naive, matmul_size, seed);
    matmul_args matmul_args_student = matmul_args_naive;
    std::println("\tMatmul: matrix size={}x{}", matmul_size, matmul_size);

    std::vector<bench_t> benchmarks = {
                // {"ReLU (Naive)",
                //  naive_relu_wrapper,
                //  naive_relu_wrapper,
                //  relu_check,
                //  &relu_args_naive,
                //  &relu_args_naive,
                //  BASELINE_RELU},
                {"ReLU (Optimized)",
                 stu_relu_wrapper,
                 naive_relu_wrapper,
                 relu_check,
                 &relu_args_student,
                 &relu_args_naive,
                 BASELINE_RELU},
                // {"Bitwise (Naive)",
                // naive_bitwise_wrapper,
                // naive_bitwise_wrapper,
                // bitwise_check,
                // &bitwise_args_naive,
                // &bitwise_args_naive,
                // BASELINE_BITWISE},
                {"Bitwise (Optimized)",
                stu_bitwise_wrapper,
                naive_bitwise_wrapper,
                bitwise_check,
                &bitwise_args_student,
                &bitwise_args_naive,
                BASELINE_BITWISE},

                {"Matmul (Optimized)",
                    stu_matmul_wrapper,
                    naive_matmul_wrapper,
                    matmul_check,
                    &matmul_args_student,
                    &matmul_args_naive,
                    BASELINE_MATMUL}
    };
    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << "\n";
    std::cout << "--------------------------------------------------------\n";

    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};
        const int k_best = 20;

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });

            avg_time += elapsed;
            debug_log("\tDEBUG: {}-th measurement: {} ns\n",
                      i,
                      static_cast<std::uint64_t>(elapsed.count()));
        }
        avg_time /= static_cast<uint64_t>(k_best);

        bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << "\n";
            std::cout
                << "  Error: Results do not match naive implementation!\n";
        } else {
            std::cout << "\033[1;32mPASSED\033[0m" << std::right
                      << std::setw(15) << avg_time.count() << " ns";
            if (avg_time.count() > bench.baseline_time.count() * 1.1) {
                std::cout << " (SLOW)";
            }
            std::cout << "\n";
        }
    }

    return 0;
}
