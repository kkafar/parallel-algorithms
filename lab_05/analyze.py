import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from dataclasses import dataclass
from polars import DataFrame, col, Config, read_csv

COL_PROCESS_COUNT = 'process_count'
COL_PROBLEM_SIZE = 'problem_size'
COL_SERIES_ID = 'series_id'
COL_TIME = 'time'
COL_TIME_AVG = COL_TIME + '_avg'
COL_TIME_STD = COL_TIME + '_std'
COL_SPEEDUP = 'speedup'
COL_SPEEDUP_AVG = COL_SPEEDUP + '_avg'
COL_SPEEDUP_STD = COL_SPEEDUP + '_std'
COL_EFFECTIVENES = 'effectivenes'
COL_EFFECTIVENES_AVG = COL_EFFECTIVENES + '_avg'
COL_EFFECTIVENES_STD = COL_EFFECTIVENES + '_std'
COL_KF = 'karp_flatt'
COL_KF_AVG = COL_KF + '_avg'
COL_KF_STD = COL_KF + '_std'
COL_COMPUTE_TIME = 'compute_time'
COL_GATHER_TIME = 'gather_time'


@dataclass
class Args:
    input_file: Path


def build_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=Path, required=True, dest='input_file', help='Path to input file')
    return parser


def get_single_cpu_time(df: DataFrame) -> DataFrame:
    problem_sizes = (
        df.lazy()
        .filter((col(COL_PROCESS_COUNT) == 1))
        .group_by(col(COL_PROBLEM_SIZE))
        .agg([
            col(COL_TIME).mean().alias(COL_TIME_AVG)
        ])
        .sort(COL_PROBLEM_SIZE)
        .collect()
    )
    return problem_sizes


def main():
    args: Args = build_cli().parse_args()
    Config.set_tbl_rows(100)
    Config.set_tbl_cols(20)
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams["errorbar.capsize"] = 2

    # process_count,problem_size,series_id,time
    df_main = read_csv(args.input_file, has_header=True).sort([col(COL_PROCESS_COUNT), col(COL_PROBLEM_SIZE)])
    df_single_cpu_time = get_single_cpu_time(df_main)
    problem_sizes = df_single_cpu_time.get_column(COL_PROBLEM_SIZE)
    single_cpu_times = df_single_cpu_time.get_column(COL_TIME_AVG)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    common_plot_args = {'capthick': 1.4, 'linestyle': ''}
    markers = ['.', 'x', '^']
    colors = 'rgb'
    assert len(markers) == len(problem_sizes)

    for i, (problem_size, m, c) in enumerate(zip(problem_sizes, markers, colors)):
        df_per_size = df_main.filter(col(COL_PROBLEM_SIZE) == problem_size)
        df_per_size = (
            df_per_size.lazy()
            .with_columns([
                (single_cpu_times.item(i) / col(COL_TIME)).alias(COL_SPEEDUP)
            ])
            .with_columns([
                (col(COL_SPEEDUP) / col(COL_PROCESS_COUNT)).alias(COL_EFFECTIVENES),
                (((1 / col(COL_SPEEDUP)) - (1 / col(COL_PROCESS_COUNT))) / (1 - (1 / col(COL_PROCESS_COUNT)))).alias(COL_KF)
            ])
            .group_by([col(COL_PROCESS_COUNT), col(COL_PROBLEM_SIZE)])
            .agg([
                col(COL_TIME).mean().alias(COL_TIME_AVG),
                col(COL_TIME).std().alias(COL_TIME_STD),
                col(COL_SPEEDUP).mean().alias(COL_SPEEDUP_AVG),
                col(COL_SPEEDUP).std().alias(COL_SPEEDUP_STD),
                col(COL_EFFECTIVENES).mean().alias(COL_EFFECTIVENES_AVG),
                col(COL_EFFECTIVENES).std().alias(COL_EFFECTIVENES_STD),
                col(COL_KF).mean().alias(COL_KF_AVG),
                col(COL_KF).std().alias(COL_KF_STD)
            ])
            .sort(col(COL_PROCESS_COUNT))
            .collect()
        )

        ax: plt.Axes = axes[0][0]
        x_data = df_per_size.get_column(COL_PROCESS_COUNT)
        y_data = df_per_size.get_column(COL_TIME_AVG)
        y_err_data = df_per_size.get_column(COL_TIME_STD)

        ax.errorbar(x_data, y_data, yerr=y_err_data, label=f'Rozmiar: {problem_size}', marker=markers[i], color=c, **common_plot_args)
        ax.plot(x_data, [single_cpu_times.item(i) / x for x in x_data], label='y = t_0 / x', linestyle='--', color=c)
        # ax.plot(x_data, [single_cpu_times.item(i) / x for x in x_data], linestyle='--')
        ax.set(
            title='Czas wykonania w zależności od liczby procesorów',
            xlabel='Liczba procesorów',
            ylabel='Czas wykonania [ms]'
        )
        ax.grid()
        ax.legend()

        ax = axes[0][1]
        y_data = df_per_size[COL_SPEEDUP_AVG]
        y_err_data = df_per_size[COL_SPEEDUP_STD]
        ax.errorbar(x_data, y_data, yerr=y_err_data, label=f'Rozmiar: {problem_size}', marker=markers[i], color=c, **common_plot_args)
        if i == 0:
            ax.plot(x_data, [x for x in x_data], label='y = x', linestyle='--')
        ax.set(
            title='Przyśpieszenie',
            xlabel='Liczba procesorów',
            ylabel='Przyśpieszenie'
        )
        ax.grid()
        ax.legend()

        ax = axes[1][0]
        y_data = df_per_size[COL_EFFECTIVENES_AVG]
        y_err_data = df_per_size[COL_EFFECTIVENES_STD]
        ax.errorbar(x_data, y_data, yerr=y_err_data, label=f'Rozmiar: {problem_size}', marker=markers[i], color=c, **common_plot_args)
        if i == 0:
            ax.plot(x_data, [1 for _ in x_data], label='y = 1', linestyle='--')
        ax.set(
            title='Efektywność',
            xlabel='Liczba procesorów',
            ylabel='Efektywność'
        )
        ax.grid()
        ax.legend()

        ax = axes[1][1]
        y_data = df_per_size[COL_KF_AVG]
        y_err_data = df_per_size[COL_KF_STD]
        ax.errorbar(x_data, y_data, yerr=y_err_data, label=f'Rozmiar: {problem_size}', marker=markers[i], color=c, **common_plot_args)
        if i == 0:
            ax.plot(x_data, [0 for _ in x_data], label='y = 0', linestyle='--')
        ax.set(
            title='Metryka Karpa-Flatta',
            xlabel='Liczba procesorów',
            ylabel='Wartość metryki'
        )
        ax.grid()
        ax.legend()

        print(df_per_size)

    fig.tight_layout()
    fig.savefig(args.input_file.parent.joinpath(args.input_file.stem).with_suffix('.png'))
    plt.show()


if __name__ == "__main__":
    main()
