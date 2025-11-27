#!/usr/bin/env python3
"""
Визуализация сходимости Hill Climbing и Beam Search.

Ожидаемые файлы:
  data/csv/hc_history.csv    (iter, score, accuracy, f1, latency)
  data/csv/beam_history.csv  (iter, score, accuracy, f1, latency)

Результат:
  data/png/hc_convergence.png
  data/png/beam_convergence.png
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data" / "csv"
PNG_DIR = ROOT / "data" / "png"
PNG_DIR.mkdir(parents=True, exist_ok=True)


def plot_convergence(csv_path: Path, title: str, out_path: Path):
    if not csv_path.exists():
        print(f"[WARN] Файл {csv_path} не найден. Пропускаю.")
        return

    df = pd.read_csv(csv_path)
    if "iter" not in df.columns or "score" not in df.columns:
        print(f"[WARN] В {csv_path} нет колонок 'iter' и 'score'. Пропускаю.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["iter"], df["score"], marker="o")
    plt.xlabel("Итерация")
    plt.ylabel("Значение целевой функции (score)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Сохранил график: {out_path}")


def main():
    plot_convergence(
        CSV_DIR / "hc_history.csv",
        "Hill Climbing: сходимость по итерациям",
        PNG_DIR / "hc_convergence.png",
    )
    plot_convergence(
        CSV_DIR / "beam_history.csv",
        "Beam Search: сходимость по итерациям",
        PNG_DIR / "beam_convergence.png",
    )


if __name__ == "__main__":
    main()
