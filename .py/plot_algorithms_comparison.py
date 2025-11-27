#!/usr/bin/env python3
"""
Сравнение итоговых результатов HC, Beam Search и SA.

Ожидаемый файл:
  data/csv/summary.csv
    (algorithm, lr, depth, reg, accuracy, f1, latency, score)

Результат:
  data/png/algorithms_score.png
  data/png/algorithms_metrics.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data" / "csv"
PNG_DIR = ROOT / "data" / "png"
PNG_DIR.mkdir(parents=True, exist_ok=True)


def plot_score(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.bar(df["algorithm"], df["score"])
    plt.xlabel("Алгоритм")
    plt.ylabel("Значение целевой функции (score)")
    plt.title("Сравнение алгоритмов по целевой функции")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Сохранил график: {out_path}")


def plot_metrics(df: pd.DataFrame, out_path: Path):
    # три метрики: accuracy, f1, latency (latency лучше отнормировать)
    plt.figure(figsize=(7, 5))

    x = range(len(df))
    width = 0.25

    plt.bar([i - width for i in x], df["accuracy"], width, label="accuracy")
    plt.bar(x, df["f1"], width, label="F1")
    # latency инвертируем (чем меньше, тем лучше) для визуального сравнения
    max_lat = df["latency"].max()
    inv_latency = max_lat - df["latency"]
    plt.bar([i + width for i in x], inv_latency, width, label="(max_latency - latency)")

    plt.xticks(list(x), df["algorithm"])
    plt.xlabel("Алгоритм")
    plt.ylabel("Значение метрик (условное)")
    plt.title("Сравнение accuracy / F1 / latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Сохранил график: {out_path}")


def main():
    csv_path = CSV_DIR / "summary.csv"
    if not csv_path.exists():
        print(f"[WARN] Файл {csv_path} не найден.")
        return

    df = pd.read_csv(csv_path)
    required = {"algorithm", "accuracy", "f1", "latency", "score"}
    if not required.issubset(df.columns):
        print(f"[WARN] В {csv_path} должны быть колонки {required}.")
        return

    plot_score(df, PNG_DIR / "algorithms_score.png")
    plot_metrics(df, PNG_DIR / "algorithms_metrics.png")


if __name__ == "__main__":
    main()
