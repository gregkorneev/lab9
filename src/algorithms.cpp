#include "hyperparams.h"
#include <cmath>

// Генерация гиперпараметров и соседей

HyperParams random_hyperparams(std::mt19937& rng, const Bounds& b) {
    std::uniform_real_distribution<double> lr_d(b.lr_min, b.lr_max);
    std::uniform_int_distribution<int>     depth_d(b.depth_min, b.depth_max);
    std::uniform_real_distribution<double> reg_d(b.reg_min, b.reg_max);
    return { lr_d(rng), depth_d(rng), reg_d(rng) };
}

HyperParams local_neighbor(const HyperParams& h,
                           std::mt19937& rng,
                           const Bounds& b,
                           double step_scale) {
    HyperParams n = h;
    std::normal_distribution<double> d(0.0, step_scale);

    n.lr  = clampT(n.lr  + d(rng) * 0.02, b.lr_min,  b.lr_max);
    n.reg = clampT(n.reg + d(rng) * 0.01, b.reg_min, b.reg_max);

    int delta_depth = static_cast<int>(std::round(d(rng) * 2.0));
    n.depth = clampT(n.depth + delta_depth, b.depth_min, b.depth_max);

    return n;
}

std::vector<HyperParams> generate_neighbors(const HyperParams& h,
                                            int k,
                                            std::mt19937& rng,
                                            const Bounds& b) {
    std::vector<HyperParams> res;
    res.reserve(k);
    for (int i = 0; i < k; ++i) {
        res.push_back(local_neighbor(h, rng, b));
    }
    return res;
}

// --------------------- Hill Climbing ---------------------- //

HyperParams hill_climbing(const HyperParams& start,
                          const Bounds& bounds,
                          std::mt19937& rng,
                          int max_iterations,
                          int neighbors_per_step) {
    HyperParams current = start;
    Metrics curM = evaluate_model(current);
    double curScore = score_for_HC(curM);

    for (int iter = 0; iter < max_iterations; ++iter) {
        HyperParams bestNeighbor = current;
        double bestScore = curScore;

        auto neighbors = generate_neighbors(current, neighbors_per_step, rng, bounds);
        for (const auto& n : neighbors) {
            Metrics m = evaluate_model(n);
            double s = score_for_HC(m);
            if (s > bestScore) {
                bestScore    = s;
                bestNeighbor = n;
            }
        }

        if (bestScore <= curScore) {
            std::cout << "[HC] остановка на итерации " << iter
                      << " — достигнут локальный максимум\n";
            break;
        }

        current  = bestNeighbor;
        curScore = bestScore;
    }
    return current;
}

// --------------------- Beam Search ---------------------- //

HyperParams beam_search(const HyperParams& start,
                        const Bounds& bounds,
                        std::mt19937& rng,
                        int beam_width,
                        int depth,
                        int neighbors_per_state) {
    std::vector<HyperParams> beam;
    beam.push_back(start);

    HyperParams globalBest = start;
    double globalBestScore = score_for_beam(evaluate_model(start));

    for (int level = 0; level < depth; ++level) {
        std::vector<std::pair<double, HyperParams>> candidates;

        for (const auto& state : beam) {
            auto neigh = generate_neighbors(state, neighbors_per_state, rng, bounds);
            for (const auto& n : neigh) {
                double score = score_for_beam(evaluate_model(n));
                candidates.push_back({score, n});
            }
        }

        if (candidates.empty()) break;

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                      return a.first > b.first;
                  });

        beam.clear();
        for (int i = 0; i < beam_width && i < (int)candidates.size(); ++i) {
            beam.push_back(candidates[i].second);
            if (candidates[i].first > globalBestScore) {
                globalBestScore = candidates[i].first;
                globalBest      = candidates[i].second;
            }
        }
    }

    return globalBest;
}

// --------------------- Имитация отжига ---------------------- //

HyperParams simulated_annealing(const HyperParams& start,
                                const Bounds& bounds,
                                std::mt19937& rng,
                                int max_iterations,
                                double T_start,
                                double T_end,
                                double alpha) {
    HyperParams current = start;
    Metrics curM = evaluate_model(current);
    double curScore = score_for_SA(curM);

    HyperParams best = current;
    double bestScore = curScore;

    double T = T_start;

    for (int t = 0; t < max_iterations && T > T_end; ++t) {
        HyperParams next = local_neighbor(current, rng, bounds, 0.4); // крупнее шаг
        Metrics nextM = evaluate_model(next);
        double nextScore = score_for_SA(nextM);

        double dE = curScore - nextScore; // максимизируем score

        if (dE < 0) {
            current  = next;
            curScore = nextScore;
        } else {
            double prob = std::exp(-dE / T);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (u(rng) < prob) {
                current  = next;
                curScore = nextScore;
            }
        }

        if (curScore > bestScore) {
            bestScore = curScore;
            best      = current;
        }

        T *= alpha; // schedule(t)
    }

    return best;
}
