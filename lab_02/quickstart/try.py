#!/usr/bin/env python
  
import mpi4py

from mpi4py import MPI

import socket

comm = MPI.COMM_WORLD

size = comm.Get_size()

rank = comm.Get_rank()

print('size=%d, rank=%d' % (size, rank))

print("hello world")

print("my rank is: %d, out of %d processes at node %s"%(rank, size, socket.gethostname()))

