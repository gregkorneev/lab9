#!/usr/bin/env python3
"""
Визуализация процесса имитации отжига.

Ожидаемый файл:
  data/csv/sa_history.csv (iter, T, score[, accepted_worse])

Результат:
  data/png/sa_temperature_score.png              — обычное сглаживание
  data/png/sa_temperature_score_smooth.png       — более сглаженная версия
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "csv" / "sa_history.csv"
PNG_DIR = ROOT / "data" / "png"


def make_plot(df: pd.DataFrame,
              score_column: str,
              out_path: Path,
              title_suffix: str):
    """Рисует один график для заданного столбца score_*."""
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Температура (левая ось)
    line_T, = ax1.plot(df["iter"], df["T"], linewidth=2, label="Температура T")
    ax1.set_xlabel("Итерация")
    ax1.set_ylabel("Температура T")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # score (правая ось)
    ax2 = ax1.twinx()
    line_S, = ax2.plot(
        df["iter"],
        df[score_column],
        linewidth=2,
        label="Целевая функция (score, сглаженный)",
    )
    ax2.set_ylabel("Значение целевой функции (score)")

    plt.title(
        f"Имитация отжига: температура и качество по итерациям ({title_suffix})",
        fontsize=14,
    )

    # Общая легенда
    lines = [line_T, line_S]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Сохранил график: {out_path}")


def main():
    if not CSV_PATH.exists():
        print(f"[WARN] Файл {CSV_PATH} не найден.")
        return

    df = pd.read_csv(CSV_PATH)
    required = {"iter", "T", "score"}
    if not required.issubset(df.columns):
        print(f"[WARN] В {CSV_PATH} должны быть колонки {required}.")
        return

    n = len(df)

    # Обычное сглаживание (как раньше)
    window_normal = 20 if n > 20 else max(3, n // 5 or 1)
    df["score_smooth"] = df["score"].rolling(window=window_normal).mean()

    # Более сильное сглаживание
    window_strong = max(50, n // 10) if n > 100 else max(10, n // 3 or 1)
    df["score_smooth_strong"] = df["score"].rolling(window=window_strong).mean()

    # Убираем NaN в начале, чтобы линии начинались не с дыр
    df_norm = df.dropna(subset=["score_smooth"]).reset_index(drop=True)
    df_strong = df.dropna(subset=["score_smooth_strong"]).reset_index(drop=True)

    # 1) Обычная версия
    make_plot(
        df_norm,
        "score_smooth",
        PNG_DIR / "sa_temperature_score.png",
        "умеренное сглаживание",
    )

    # 2) Более сглаженная версия
    make_plot(
        df_strong,
        "score_smooth_strong",
        PNG_DIR / "sa_temperature_score_smooth.png",
        "сильное сглаживание",
    )


if __name__ == "__main__":
    main()
