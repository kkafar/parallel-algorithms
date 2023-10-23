#!/usr/bin/env python

import numpy as np
import sys
from mpi4py import MPI


def compute_primes_up_to(upper_bound: int):
    numbers = np.arange(upper_bound + 1, dtype=int)
    mask = np.ones_like(numbers, dtype=bool)
    mask[0:2] = False
    for i in range(2, int(np.sqrt(upper_bound)) + 1):
        if mask[i]:
            mask[i * i::i] = False
    return numbers[mask]


def compute_primes_in_interval(lower: int, upper: int, factors: list):
    numbers = np.arange(lower, upper + 1, dtype=int)
    mask = np.ones_like(numbers, dtype=bool)
    for n in numbers:
        i = n - lower
        for d in factors:
            if n % d == 0:
                mask[i] = False
                break
    return numbers[mask]


def master(comm, size, limit_B, upper_bound):
    # Point 2
    # Domain decomposition
    last_worker_rank = size - 1
    interval_length = int(np.ceil((upper_bound - limit_B) / last_worker_rank))

    # Send tasks to workers
    lower_i = limit_B + 1
    upper_i = limit_B + interval_length
    for i in range(1, last_worker_rank):
        comm.send({'lower': lower_i, 'upper': upper_i}, dest=i)
        lower_i += interval_length
        upper_i += interval_length
    # Last process might have less to do
    comm.send({'lower': lower_i, 'upper': upper_bound}, dest=last_worker_rank)


def worker(comm, factors_B, master_rank):
    # Point 3
    message = comm.recv(source=master_rank)
    lower_bound, upper_bound = message['lower'], message['upper']
    computed_primes = compute_primes_in_interval(lower_bound, upper_bound, factors_B)
    comm.send(computed_primes, dest=master_rank)


def compute_primes(upper_bound: int):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set B was defined in task description
    limit_B = int(np.sqrt(upper_bound))
    factors_B = compute_primes_up_to(limit_B)
    primes = np.copy(factors_B)

    if rank == 0:
        master(comm, size, limit_B, upper_bound)
    else:
        worker(comm, factors_B, 0)

    if rank == 0:
        # Point 4
        for i in range(1, size):
            received_primes = comm.recv(source=i)
            primes = np.concatenate((primes, received_primes))

        n_primes = len(primes)
        print(f'Found {n_primes} for upper bound: {upper_bound}')
        # Sort copies the array
        print(np.sort(primes))

    MPI.Finalize()


def main():
    upper_bound = int(sys.argv[1])
    compute_primes(upper_bound)


if __name__ == "__main__":
    main()
