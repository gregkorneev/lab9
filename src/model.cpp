#include "hyperparams.h"
#include <cmath>

// вывод структур
std::ostream& operator<<(std::ostream& os, const HyperParams& h) {
    os << "{lr=" << h.lr
       << ", глубина=" << h.depth
       << ", рег=" << h.reg << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Metrics& m) {
    os << "{accuracy=" << m.accuracy
       << ", F1=" << m.f1
       << ", задержка=" << m.latency << "}";
    return os;
}

// «Модель» — аналитическая функция метрик
Metrics evaluate_model(const HyperParams& h) {
    double lr    = h.lr;
    double depth = static_cast<double>(h.depth);
    double reg   = h.reg;

    // «идеальная» область: lr≈0.05, depth≈5, reg≈0.01
    double acc = 1.0
        - 30.0 * std::pow(lr - 0.05, 2)
        - 0.03 * std::pow(depth - 5.0, 2)
        - 200.0 * std::pow(reg - 0.01, 2);

    double f1  = 1.0
        - 20.0 * std::pow(lr - 0.04, 2)
        - 0.04 * std::pow(depth - 6.0, 2)
        - 150.0 * std::pow(reg - 0.02, 2);

    // latency растёт с глубиной и уменьшается с lr
    double latency = 50.0 + 3.0 * depth + 40.0 * (0.1 - lr);

    // небольшой шум, чтобы были локальные максимумы
    double noise = 0.01 * std::sin(20.0 * lr)
                          * std::cos(0.5 * depth)
                          * std::sin(200.0 * reg);
    acc += noise;
    f1  += 0.5 * noise;

    acc     = clampT(acc, 0.0, 1.0);
    f1      = clampT(f1, 0.0, 1.0);
    latency = std::max(1.0, latency);

    return {acc, f1, latency};
}

// Целевые функции
double score_for_HC(const Metrics& m) {
    return m.accuracy;
}

double score_for_beam(const Metrics& m) {
    double normAcc = m.accuracy;
    double normF1  = m.f1;
    double normLat = 1.0 / (1.0 + m.latency / 100.0); // меньше latency -> выше значение

    return 0.5 * normAcc + 0.3 * normF1 + 0.2 * normLat;
}

double score_for_SA(const Metrics& m) {
    return m.accuracy;
}
