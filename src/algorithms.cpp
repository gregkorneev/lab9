#include "hyperparams.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

// ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ГЕНЕРАЦИИ ================== //

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

// ================== HILL CLIMBING (с логированием в CSV) ================== //

HyperParams hill_climbing(const HyperParams& start,
                          const Bounds& bounds,
                          std::mt19937& rng,
                          int max_iterations,
                          int neighbors_per_step) {
    // Подготовка CSV: data/csv/hc_history.csv
    fs::path csvDir = fs::path("data") / "csv";
    fs::create_directories(csvDir);
    fs::path hcPath = csvDir / "hc_history.csv";
    std::ofstream hcOut(hcPath);
    if (!hcOut) {
        std::cerr << "[HC] Не удалось открыть " << hcPath << " для записи\n";
    } else {
        hcOut << "iter,score,accuracy,f1,latency\n";
    }

    HyperParams current = start;
    Metrics curM = evaluate_model(current);
    double curScore = score_for_HC(curM);

    // Логируем начальное состояние (итерация 0)
    if (hcOut) {
        hcOut << 0 << ","
              << curScore << ","
              << curM.accuracy << ","
              << curM.f1 << ","
              << curM.latency << "\n";
    }

    for (int iter = 1; iter <= max_iterations; ++iter) {
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
        curM     = evaluate_model(current);
        curScore = score_for_HC(curM);

        if (hcOut) {
            hcOut << iter << ","
                  << curScore << ","
                  << curM.accuracy << ","
                  << curM.f1 << ","
                  << curM.latency << "\n";
        }
    }

    return current;
}

// ================== BEAM SEARCH (с логированием в CSV) ================== //

HyperParams beam_search(const HyperParams& start,
                        const Bounds& bounds,
                        std::mt19937& rng,
                        int beam_width,
                        int depth,
                        int neighbors_per_state) {
    // Подготовка CSV: data/csv/beam_history.csv
    fs::path csvDir = fs::path("data") / "csv";
    fs::create_directories(csvDir);
    fs::path beamPath = csvDir / "beam_history.csv";
    std::ofstream beamOut(beamPath);
    if (!beamOut) {
        std::cerr << "[Beam] Не удалось открыть " << beamPath << " для записи\n";
    } else {
        beamOut << "iter,score,accuracy,f1,latency\n";
    }

    std::vector<HyperParams> beam;
    beam.push_back(start);

    HyperParams globalBest = start;
    Metrics globalBestM = evaluate_model(start);
    double globalBestScore = score_for_beam(globalBestM);

    // Лог начального состояния (итерация 0)
    if (beamOut) {
        beamOut << 0 << ","
                << globalBestScore << ","
                << globalBestM.accuracy << ","
                << globalBestM.f1 << ","
                << globalBestM.latency << "\n";
    }

    for (int level = 1; level <= depth; ++level) {
        std::vector<std::pair<double, HyperParams>> candidates;

        for (const auto& state : beam) {
            auto neigh = generate_neighbors(state, neighbors_per_state, rng, bounds);
            for (const auto& n : neigh) {
                double score = score_for_beam(evaluate_model(n));
                candidates.push_back({score, n});
            }
        }

        if (candidates.empty()) {
            break;
        }

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
                globalBestM     = evaluate_model(globalBest);
            }
        }

        // Логируем лучшего на текущем уровне (по глобальному scору)
        if (beamOut) {
            beamOut << level << ","
                    << globalBestScore << ","
                    << globalBestM.accuracy << ","
                    << globalBestM.f1 << ","
                    << globalBestM.latency << "\n";
        }
    }

    return globalBest;
}

// ================== SIMULATED ANNEALING (с логированием в CSV) =========== //

HyperParams simulated_annealing(const HyperParams& start,
                                const Bounds& bounds,
                                std::mt19937& rng,
                                int max_iterations,
                                double T_start,
                                double T_end,
                                double alpha) {
    // Подготовка CSV: data/csv/sa_history.csv
    fs::path csvDir = fs::path("data") / "csv";
    fs::create_directories(csvDir);
    fs::path saPath = csvDir / "sa_history.csv";
    std::ofstream saOut(saPath);
    if (!saOut) {
        std::cerr << "[SA] Не удалось открыть " << saPath << " для записи\n";
    } else {
        saOut << "iter,T,score,accepted_worse\n";
    }

    HyperParams current = start;
    Metrics curM = evaluate_model(current);
    double curScore = score_for_SA(curM);

    HyperParams best = current;
    double bestScore = curScore;

    double T = T_start;

    // Лог начального состояния (итерация 0)
    if (saOut) {
        saOut << 0 << ","
              << T << ","
              << curScore << ","
              << 0 << "\n";
    }

    for (int t = 1; t <= max_iterations && T > T_end; ++t) {
        HyperParams next = local_neighbor(current, rng, bounds, 0.4); // крупнее шаг
        Metrics nextM = evaluate_model(next);
        double nextScore = score_for_SA(nextM);

        double dE = curScore - nextScore; // максимизируем score

        bool acceptedWorse = false;

        if (dE < 0) {
            // лучшее решение
            current  = next;
            curM     = nextM;
            curScore = nextScore;
        } else {
            double prob = std::exp(-dE / T);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (u(rng) < prob) {
                current  = next;
                curM     = nextM;
                curScore = nextScore;
                acceptedWorse = true;
            }
        }

        if (curScore > bestScore) {
            bestScore = curScore;
            best      = current;
        }

        // Лог на текущей итерации
        if (saOut) {
            saOut << t << ","
                  << T << ","
                  << curScore << ","
                  << (acceptedWorse ? 1 : 0) << "\n";
        }

        T *= alpha; // охлаждение
    }

    return best;
}
