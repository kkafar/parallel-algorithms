import argparse
import numpy as np
from typing import NamedTuple


class Args(NamedTuple):
    cities: int
    radius: float


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', type=int, required=True, help='Size of the problem')
    parser.add_argument('--radius', type=float, required=True, help='Radius of the circle we are sampling cities from')


def main():
    args = build_cli().parse_args()


if __name__ == "__main__":
    main()
