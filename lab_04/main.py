#!/usr/bin/env python

import numpy as np
import argparse
from mpi4py import MPI
from timeit import default_timer as timer
from dataclasses import dataclass


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solver")
    parser.add_argument('-s', '--series', type=int, required=True, dest='series', help='Id of the series. THIS DOES NOT MEAN COMPUTATION WILL BE REPEATED')
    parser.add_argument('--points-per-proc', type=int, required=True, dest='ppc', help='Resolution for each process')
    parser.add_argument('-a', '--side', type=float, required=True, dest='a', help='Side length of the membrane')
    parser.add_argument('--theta', type=float, required=True, dest='theta', help='Right side of the equation')
    parser.add_argument('--iters', type=int, required=True, dest='iters', help='Number of iterations in iterative method')
    return parser


@dataclass
class Args:
    series: int
    ppc: int
    a: float
    theta: float
    iters: int


def compute(comm, rank, size, delta, ppc, theta, iters):
    """
    :param delta: resolution of the grid
    """

    side = rank * ppc
    # Rectangle this process is responsible for
    H = np.zeros((ppc, side))

    x_min = 0 if rank > 0 else 1
    x_max = ppc if rank < size - 1 else ppc - 1

    for i in range(iters):
        H_i = np.zeros((ppc, side), dtype=np.float64)
        if i > 0:
            # We receive values from last iteration from our neighs
            if rank > 0:
                recv_buff = np.empty(side, dtype=np.float64)
                comm.Recv(recv_buff, rank - 1, i - 1)
                H_i[0] += recv_buff
            if rank < size - 1:
                recv_buff = np.empty(side, dtype=np.float64)
                comm.Recv(recv_buff, rank + 1, i - 1)
                H_i[-1] += recv_buff

        # We apply the formula
        H_i[x_min: x_max, 1: side - 1] -= delta ** 2 * theta
        H_i[x_min:x_max, 1: side - 1] += H[x_min:x_max, 0: side - 2]
        H_i[x_min:x_max, 1: side - 1] += H[x_min:x_max, 2:side]
        H_i[1:x_max, :] += H[0: x_max - 1, :]
        H_i[x_min: ppc - 1, :] += H[x_min + 1: ppc, :]

        H_i /= 4
        H = H_i

        if rank > 0:
            comm.Isend(H[0].copy(), rank - 1, i)

        if rank < size - 1:
            comm.Isend(H[-1].copy(), rank + 1, i)

    return H


def main():
    args: Args = build_cli().parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Hey, I'm not counting in argument parsing as sequential part of the program,
    # as it could be completely avoided and is done only for convenience
    start_time = timer()
    delta = args.a / (args.ppc * size - 1)
    stride = compute(comm, rank, size, delta, args.ppc, args.theta, args.iters)
    recv_buff = comm.gather(stride, root=0)
    if rank == 0:
        recv_buff = np.concatenate(recv_buff, axis=0)
        elapsed = timer() - start_time  # Result is in seconds, we want to convert it to milis
        # "process_count,problem_size,series_id,time"
        print(f'{size},{args.ppc},{args.series},{elapsed * 1000}')


if __name__ == "__main__":
    main()
