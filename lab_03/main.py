import sys
import numpy as np
from mpi4py import MPI


def compute(comm, proc_rank, n_proc, delta, ppc, theta, steps):
    """
    :param comm: MPI global communicator
    :param proc_rank: rank of current process
    :param n_proc: total number of processes
    :param delta: ?
    :param ppc: points per process
    :theta: ?
    :steps: ?
    """

    side = proc_rank * ppc
    stride = np.zeros((ppc, side))

    x_min = 0 if proc_rank > 0 else 1
    x_max = ppc

    if proc_rank >= n_proc:
        x_max -= 1

    for s in range(steps):
        new_stride = np.zeros((ppc, side), dtype=np.float64)
        if s > 0:
            if proc_rank > 0:
                recv_buff = np.empty(side, dtype=np.float64)
                comm.Recv(recv_buff, proc_rank - 1, s - 1)
                new_stride[0] += recv_buff
            if proc_rank < n_proc - 1:
                recv_buff = np.empty(side, dtype=np.float64)
                comm.Recv(recv_buff, proc_rank + 1, s - 1)
                new_stride[-1] += recv_buff

        new_stride[x_min: x_max, 1: side - 1] -= delta ** 2 * theta
        new_stride[x_min:x_max, 1: side - 1] += stride[x_min:x_max, 0: side - 2]
        new_stride[x_min:x_max, 1: side - 1] += stride[x_min:x_max, 2:side]
        new_stride[1:x_max, :] += stride[0: x_max - 1, :]
        new_stride[x_min: ppc - 1, :] += stride[x_min + 1: ppc, :]

        new_stride /= 4
        stride = new_stride

        if proc_rank > 0:
            comm.Isend(stride[0].copy(), proc_rank - 1, s)

        if proc_rank < n_proc - 1:
            comm.Isend(stride[-1].copy(), proc_rank + 1, s)

    return stride


# def main():
#     pass

if __name__ == "__main__":
    _, ppc, a, theta, steps = sys.argv
    ppc = int(ppc)
    a = float(a)
    theta = float(theta)
    steps = int(steps)

    comm = MPI.COMM_WORLD
    proc_rank = comm.Get_rank()
    n_proc = comm.Get_size()

    delta = a / (ppc * n_proc - 1)
    stride = compute(comm, proc_rank, n_proc, delta, ppc, theta, steps)

    recv_buff = comm.gather(stride, root=0)
    if proc_rank == 0:
        recv_buff = np.concatenate(recv_buff, axis=0)
        np.savetxt(sys.stdout.buffer, recv_buff)


