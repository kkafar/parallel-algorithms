#!/usr/bin/env python

import sys
import numpy as np
from mpi4py import MPI

class Args:
    def __init__(self):
        assert len(sys.argv) == 5
        # Points per process
        self.ppc: int = sys.argv[1]
        # Membrane side length
        self.a: float = sys.argv[2]
        self.theta: float = sys.argv[3]
        # Iteration count of the Jacobi method
        self.iters: int = sys.argv[4]

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
            recv_buff = np.empty(side, dtype=np.float64)
            if rank > 0:
                comm.Recv(recv_buff, rank - 1, i - 1)
                H_i[0] += recv_buff
            if rank < size - 1:
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
    args = Args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    delta = args.a / (args.ppc * size - 1)
    stride = compute(comm, rank, size, delta, args.ppc, args.theta, args.iters)
    recv_buff = comm.gather(stride, root=0)
    if rank == 0:
        recv_buff = np.concatenate(recv_buff, axis=0)
        np.savetxt(sys.stdout.buffer, recv_buff)

if __name__ == "__main__":
    main()
