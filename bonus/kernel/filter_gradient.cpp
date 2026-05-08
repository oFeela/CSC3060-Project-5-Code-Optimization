#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#include <thread>
#include "helpers.h"

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count);
    args->data.b.resize(count);
    args->data.c.resize(count);
    args->data.d.resize(count);
    args->data.e.resize(count);
    args->data.f.resize(count);
    args->data.g.resize(count);
    args->data.h.resize(count);
    args->data.i.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen);
        args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen);
        args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen);
        args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen);
        args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);
    }
}

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    double total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {

            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += data.a[idx];
                    sum_b += data.b[idx];
                    sum_c += data.c[idx];
                }
            }
            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;

            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx =
                -data.d[ym1 + xm1] + data.d[ym1 + xp1]
                -2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1]
                -data.d[yp1 + xm1] + data.d[yp1 + xp1];

            const float sobel_ex =
                -data.e[ym1 + xm1] + data.e[ym1 + xp1]
                -2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1]
                -data.e[yp1 + xm1] + data.e[yp1 + xp1];

            const float sobel_fx =
                -data.f[ym1 + xm1] + data.f[ym1 + xp1]
                -2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1]
                -data.f[yp1 + xm1] + data.f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -data.g[ym1 + xm1] - 2.0f * data.g[ym1 + x0] - data.g[ym1 + xp1]
                + data.g[yp1 + xm1] + 2.0f * data.g[yp1 + x0] + data.g[yp1 + xp1];

            const float sobel_hy =
                -data.h[ym1 + xm1] - 2.0f * data.h[ym1 + x0] - data.h[ym1 + xp1]
                + data.h[yp1 + xm1] + 2.0f * data.h[yp1 + x0] + data.h[yp1 + xp1];

            const float sobel_iy =
                -data.i[ym1 + xm1] - 2.0f * data.i[ym1 + x0] - data.i[ym1 + xp1]
                + data.i[yp1 + xm1] + 2.0f * data.i[yp1 + x0] + data.i[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = total;
}

void convert_soa_to_aos(std::vector<pixel> &new_data, const data_struct &old_data)
{
    std::size_t count = old_data.a.size();
    new_data.resize(count);
    for (std::size_t i = 0; i < count; ++i){
        new_data[i] = pixel(
            old_data.a[i], old_data.b[i], old_data.c[i], old_data.d[i],
            old_data.e[i], old_data.f[i], old_data.g[i], old_data.h[i], old_data.i[i]
        );
    }
}

void stu_filter_gradient(float& out, const std::vector<pixel>& data,
                   std::size_t width, std::size_t height) {
    
    // modified to use the thread wrapper helper
    #if 1
    const std::size_t H = height;
    const std::size_t W = width;
    constexpr float inv9 = 1.0f / 9.0f;
    std::atomic<double> total(0.0);

    parallel_for(1, H - 1, [&](std::size_t st_y, std::size_t en_y) {
        double local_sum = 0.0;

        for (std::size_t y = st_y; y < en_y; ++y) {
            for (std::size_t x = 1; x + 1 < W; ++x) {
                double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const std::size_t row = (y + dy) * W;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const std::size_t idx = row + (x + dx);
                        sum_a += data[idx].a;
                        sum_b += data[idx].b;
                        sum_c += data[idx].c;
                    }
                }
                const float avg_a = sum_a * inv9;
                const float avg_b = sum_b * inv9;
                const float avg_c = sum_c * inv9;
                const float p1 = avg_a * avg_b + avg_c;

                const std::size_t ym1 = (y - 1) * W;
                const std::size_t y0 = y * W;
                const std::size_t yp1 = (y + 1) * W;

                const std::size_t xm1 = x - 1;
                const std::size_t x0 = x;
                const std::size_t xp1 = x + 1;

                const pixel &p_ym1_xm1 = data[ym1 + xm1];
                const pixel &p_ym1_x0 = data[ym1 + x0];
                const pixel &p_ym1_xp1 = data[ym1 + xp1];
                const pixel &p_y0_xm1 = data[y0 + xm1];
                const pixel &p_y0_xp1 = data[y0 + xp1];
                const pixel &p_yp1_xm1 = data[yp1 + xm1];
                const pixel &p_yp1_x0 = data[yp1 + x0];
                const pixel &p_yp1_xp1 = data[yp1 + xp1];

                const float sobel_dx =
                    -p_ym1_xm1.d + p_ym1_xp1.d - 2.0f * p_y0_xm1.d +
                    2.0f * p_y0_xp1.d - p_yp1_xm1.d + p_yp1_xp1.d;

                const float sobel_ex =
                    -p_ym1_xm1.e + p_ym1_xp1.e - 2.0f * p_y0_xm1.e +
                    2.0f * p_y0_xp1.e - p_yp1_xm1.e + p_yp1_xp1.e;

                const float sobel_fx =
                    -p_ym1_xm1.f + p_ym1_xp1.f - 2.0f * p_y0_xm1.f +
                    2.0f * p_y0_xp1.f - p_yp1_xm1.f + p_yp1_xp1.f;

                const float p2 = sobel_dx * sobel_ex + sobel_fx;

                const float sobel_gy = -p_ym1_xm1.g - 2.0f * p_ym1_x0.g -
                                        p_ym1_xp1.g + p_yp1_xm1.g +
                                        2.0f * p_yp1_x0.g + p_yp1_xp1.g;

                const float sobel_hy = -p_ym1_xm1.h - 2.0f * p_ym1_x0.h -
                                        p_ym1_xp1.h + p_yp1_xm1.h +
                                        2.0f * p_yp1_x0.h + p_yp1_xp1.h;

                const float sobel_iy = -p_ym1_xm1.i - 2.0f * p_ym1_x0.i -
                                        p_ym1_xp1.i + p_yp1_xm1.i +
                                        2.0f * p_yp1_x0.i + p_yp1_xp1.i;

                const float p3 = sobel_gy * sobel_hy + sobel_iy;

                local_sum += p1 + p2 + p3;
            }
        }
        total += local_sum;
    });

    out = total;
    #endif

    // multithread, original, written by Geo
    #if 0
    const std::size_t H = height;
    const std::size_t W = width;
    constexpr float inv9 = 1.0f / 9.0f;

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4;

    std::vector<double> partial_sums(num_threads, 0.0);
    std::vector<std::thread> threads;

    std::size_t rows_per_thread = (H - 2) / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        std::size_t start_y = 1 + t * rows_per_thread;
        std::size_t end_y =
            (t == num_threads - 1) ? H - 1 : start_y + rows_per_thread;

        threads.emplace_back([&, t, start_y, end_y]() {
            double local_sum = 0.0;

            for (std::size_t y = start_y; y < end_y; ++y) {
                for (std::size_t x = 1; x + 1 < W; ++x) {
                    double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        const std::size_t row = (y + dy) * W;
                        for (int dx = -1; dx <= 1; ++dx) {
                            const std::size_t idx = row + (x + dx);
                            sum_a += data[idx].a;
                            sum_b += data[idx].b;
                            sum_c += data[idx].c;
                        }
                    }
                    const float avg_a = sum_a * inv9;
                    const float avg_b = sum_b * inv9;
                    const float avg_c = sum_c * inv9;
                    const float p1 = avg_a * avg_b + avg_c;

                    const std::size_t ym1 = (y - 1) * W;
                    const std::size_t y0 = y * W;
                    const std::size_t yp1 = (y + 1) * W;

                    const std::size_t xm1 = x - 1;
                    const std::size_t x0 = x;
                    const std::size_t xp1 = x + 1;

                    const pixel &p_ym1_xm1 = data[ym1 + xm1];
                    const pixel &p_ym1_x0 = data[ym1 + x0];
                    const pixel &p_ym1_xp1 = data[ym1 + xp1];
                    const pixel &p_y0_xm1 = data[y0 + xm1];
                    const pixel &p_y0_xp1 = data[y0 + xp1];
                    const pixel &p_yp1_xm1 = data[yp1 + xm1];
                    const pixel &p_yp1_x0 = data[yp1 + x0];
                    const pixel &p_yp1_xp1 = data[yp1 + xp1];

                    const float sobel_dx =
                        -p_ym1_xm1.d + p_ym1_xp1.d - 2.0f * p_y0_xm1.d +
                        2.0f * p_y0_xp1.d - p_yp1_xm1.d + p_yp1_xp1.d;

                    const float sobel_ex =
                        -p_ym1_xm1.e + p_ym1_xp1.e - 2.0f * p_y0_xm1.e +
                        2.0f * p_y0_xp1.e - p_yp1_xm1.e + p_yp1_xp1.e;

                    const float sobel_fx =
                        -p_ym1_xm1.f + p_ym1_xp1.f - 2.0f * p_y0_xm1.f +
                        2.0f * p_y0_xp1.f - p_yp1_xm1.f + p_yp1_xp1.f;

                    const float p2 = sobel_dx * sobel_ex + sobel_fx;

                    const float sobel_gy = -p_ym1_xm1.g - 2.0f * p_ym1_x0.g -
                                           p_ym1_xp1.g + p_yp1_xm1.g +
                                           2.0f * p_yp1_x0.g + p_yp1_xp1.g;

                    const float sobel_hy = -p_ym1_xm1.h - 2.0f * p_ym1_x0.h -
                                           p_ym1_xp1.h + p_yp1_xm1.h +
                                           2.0f * p_yp1_x0.h + p_yp1_xp1.h;

                    const float sobel_iy = -p_ym1_xm1.i - 2.0f * p_ym1_x0.i -
                                           p_ym1_xp1.i + p_yp1_xm1.i +
                                           2.0f * p_yp1_x0.i + p_yp1_xp1.i;

                    const float p3 = sobel_gy * sobel_hy + sobel_iy;

                    local_sum += p1 + p2 + p3;
                }
            }

            partial_sums[t] = local_sum;
        }); // end of thread
    }

    // Wait for all threads
    for (auto &th : threads) {
        th.join();
    }

    // Combine results
    double total = 0.0;
    for (auto sum : partial_sums) {
        total += sum;
    }

    out = total;
    #endif
}

void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}
void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    stu_filter_gradient(args.out, args.aos_data, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;
    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;
    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
