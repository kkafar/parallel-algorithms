#!/usr/bin/env python

import numpy as np
import argparse
from mpi4py import MPI
from timeit import default_timer as timer
from dataclasses import dataclass


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solver")
    parser.add_argument('-s', '--series', type=int, required=True, dest='series', help='Id of the series. THIS DOES NOT MEAN COMPUTATION WILL BE REPEATED')
    parser.add_argument('--grid-points', type=int, required=True, dest='grid_points', help='Number of points on the grid (single side), total would be <this_value> ^ 2')
    parser.add_argument('-a', '--side', type=float, required=True, dest='a', help='Side length of the membrane')
    parser.add_argument('--theta', type=float, required=True, dest='theta', help='Right side of the equation')
    parser.add_argument('--iters', type=int, required=True, dest='iters', help='Number of iterations in iterative method')
    return parser


@dataclass
class Args:
    series: int
    grid_points: int
    a: float
    theta: float
    iters: int


def vsize_for_rank(mpi_rank: int, mpi_size: int, grid_points: int) -> int:
    size = grid_points // mpi_size
    remainder = grid_points % mpi_size

    if remainder > 0 and mpi_rank < remainder:
        size += 1
    return size


def compute(comm, rank, size, delta, grid_points, theta, iters):
    """
    :param delta: resolution of the grid
    """

    # side = size * ppc
    hsize = grid_points
    vsize = vsize_for_rank(rank, size, grid_points)

    # Rectangle this process is responsible for
    H = np.zeros((vsize, hsize), dtype=np.float64)
    # H_i = np.zeros((ppc, side), dtype=np.float64)
    recv_buff = np.empty(hsize, dtype=np.float64)

    x_min = 0 if rank > 0 else 1
    x_max = vsize if rank < size - 1 else vsize - 1

    for i in range(iters):
        H_i = np.zeros((vsize, hsize), dtype=np.float64)
        if i > 0:
            # We receive values from last iteration from our neighs
            if rank > 0:
                comm.Recv(recv_buff, rank - 1, i - 1)
                H_i[0] += recv_buff
            if rank < size - 1:
                # recv_buff = np.zeros(side, dtype=np.float64)
                comm.Recv(recv_buff, rank + 1, i - 1)
                H_i[-1] += recv_buff

        # We apply the formula
        H_i[x_min : x_max, 1 : hsize - 1] -= delta ** 2 * theta
        H_i[x_min : x_max, 1 : hsize - 1] += H[x_min : x_max, 0 : hsize - 2]
        H_i[x_min : x_max, 1 : hsize - 1] += H[x_min : x_max, 2 : hsize]
        H_i[1 : x_max, :] += H[0 : x_max - 1, :]
        H_i[x_min : vsize - 1, :] += H[x_min + 1 : vsize, :]

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

    # print(f'rank {rank} args {args} size {size}')
    # Hey, I'm not counting in argument parsing as sequential part of the program,
    # as it could be completely avoided and is done only for convenience
    start_time = timer()

    # ppc = args.grid_points / size

    delta = args.a / (args.grid_points - 1)
    # compute_time = timer()
    stripe = compute(comm, rank, size, delta, args.grid_points, args.theta, args.iters)
    # compute_time = timer() - compute_time

    # gather_time = timer()

    # recv_buff = np.empty((args.grid_points, args.grid_points), dtype=np.float64)
    # comm.Gather(stripe, recv_buff, root=0)

    recv_buff = comm.gather(stripe, root=0)
    # gather_time = timer() - gather_time

    if rank == 0:
        assert len(recv_buff) > 0
        # recv_buff = np.concatenate(recv_buff, axis=0)
        elapsed = timer() - start_time  # Result is in seconds, we want to convert it to milis
        # "process_count,problem_size,series_id,time,compute_time,gather_time"
        # print(f'{size},{args.ppc},{args.series},{elapsed * 1000},{compute_time * 1000},{gather_time * 1000}')
        print(f'{size},{args.grid_points},{args.series},{elapsed * 1000}')


if __name__ == "__main__":
    main()
