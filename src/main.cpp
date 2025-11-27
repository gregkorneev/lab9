#include "hyperparams.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Bounds bounds;
    std::mt19937 rng(
        static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch().count()
        )
    );

    // Случайные стартовые гиперпараметры
    HyperParams start = random_hyperparams(rng, bounds);
    Metrics startM = evaluate_model(start);

    std::cout << "Стартовые гиперпараметры:  " << start
              << " -> метрики " << startM << "\n\n";

    // 1) Hill Climbing
    std::cout << "==== Hill Climbing: оптимизация метрики accuracy ====\n";
    HyperParams bestHC = hill_climbing(start, bounds, rng);
    Metrics mHC = evaluate_model(bestHC);

    std::cout << "Лучшие параметры (Hill Climbing): " << bestHC << "\n";
    std::cout << "Метрики:                          " << mHC
              << "  (значение целевой функции = " << score_for_HC(mHC) << ")\n\n";

    // 2) Beam Search
    std::cout << "==== Beam Search: баланс accuracy, F1 и времени отклика ====\n";
    HyperParams bestBeam = beam_search(start, bounds, rng, 5);
    Metrics mBeam = evaluate_model(bestBeam);

    std::cout << "Лучшие параметры (Beam Search):   " << bestBeam << "\n";
    std::cout << "Метрики:                          " << mBeam
              << "  (комбинированный скор = " << score_for_beam(mBeam) << ")\n\n";

    // 3) Имитация отжига
    std::cout << "==== Имитация отжига: исследование экстремальных значений параметров ====\n";

    HyperParams middle{
        (bounds.lr_min    + bounds.lr_max) / 2.0,
        (bounds.depth_min + bounds.depth_max) / 2,
        (bounds.reg_min   + bounds.reg_max) / 2.0
    };

    HyperParams bestSA = simulated_annealing(
        middle, bounds, rng,
        2000,   // max_iterations
        1.5,    // T_start
        1e-4,   // T_end
        0.995   // alpha
    );

    Metrics mSA = evaluate_model(bestSA);

    std::cout << "Лучшие параметры (имитация отжига): " << bestSA << "\n";
    std::cout << "Метрики:                            " << mSA
              << "  (значение целевой функции = " << score_for_SA(mSA) << ")\n";

    // ---------- ЛОГИРОВАНИЕ ИТОГОВ В CSV ---------- //
    namespace fs = std::filesystem;
    fs::path csvDir = fs::path("data") / "csv";
    fs::create_directories(csvDir);

    fs::path summaryPath = csvDir / "summary.csv";
    std::ofstream out(summaryPath);
    if (!out) {
        std::cerr << "Не удалось открыть файл " << summaryPath << " для записи\n";
        return 1;
    }

    out << "algorithm,lr,depth,reg,accuracy,f1,latency,score\n";
    out << std::fixed << std::setprecision(6);

    out << "HC,"
        << bestHC.lr << "," << bestHC.depth << "," << bestHC.reg << ","
        << mHC.accuracy << "," << mHC.f1 << "," << mHC.latency << ","
        << score_for_HC(mHC) << "\n";

    out << "Beam,"
        << bestBeam.lr << "," << bestBeam.depth << "," << bestBeam.reg << ","
        << mBeam.accuracy << "," << mBeam.f1 << "," << mBeam.latency << ","
        << score_for_beam(mBeam) << "\n";

    out << "SA,"
        << bestSA.lr << "," << bestSA.depth << "," << bestSA.reg << ","
        << mSA.accuracy << "," << mSA.f1 << "," << mSA.latency << ","
        << score_for_SA(mSA) << "\n";

    std::cout << "\n[INFO] Итоговые результаты сохранены в " << summaryPath << "\n";

    return 0;
}
