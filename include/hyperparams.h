#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

// --------------------- Общие структуры ---------------------- //

struct HyperParams {
    double lr;      // скорость обучения
    int    depth;   // «глубина» модели / число слоёв
    double reg;     // коэффициент регуляризации
};

struct Metrics {
    double accuracy;   // основная метрика
    double f1;         // дополнительная
    double latency;    // «время отклика» (чем меньше, тем лучше)
};

struct Bounds {
    double lr_min   = 0.001;
    double lr_max   = 0.10;
    int    depth_min = 1;
    int    depth_max = 10;
    double reg_min  = 0.0;
    double reg_max  = 0.05;
};

std::ostream& operator<<(std::ostream& os, const HyperParams& h);
std::ostream& operator<<(std::ostream& os, const Metrics& m);

template<typename T>
T clampT(T x, T lo, T hi) {
    return std::max(lo, std::min(hi, x));
}

// --------- Объявления функций (реализация в .cpp) ---------- //

Metrics evaluate_model(const HyperParams& h);

double score_for_HC(const Metrics& m);
double score_for_beam(const Metrics& m);
double score_for_SA(const Metrics& m);

HyperParams random_hyperparams(std::mt19937& rng, const Bounds& b);
HyperParams local_neighbor(const HyperParams& h,
                           std::mt19937& rng,
                           const Bounds& b,
                           double step_scale = 0.2);

std::vector<HyperParams> generate_neighbors(const HyperParams& h,
                                            int k,
                                            std::mt19937& rng,
                                            const Bounds& b);

HyperParams hill_climbing(const HyperParams& start,
                          const Bounds& bounds,
                          std::mt19937& rng,
                          int max_iterations = 200,
                          int neighbors_per_step = 20);

HyperParams beam_search(const HyperParams& start,
                        const Bounds& bounds,
                        std::mt19937& rng,
                        int beam_width = 5,
                        int depth = 30,
                        int neighbors_per_state = 10);

HyperParams simulated_annealing(const HyperParams& start,
                                const Bounds& bounds,
                                std::mt19937& rng,
                                int max_iterations = 1000,
                                double T_start = 1.0,
                                double T_end   = 1e-3,
                                double alpha   = 0.99);
