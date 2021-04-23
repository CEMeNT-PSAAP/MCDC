import numpy as np

from mpi4py import MPI
from math   import floor


# Communication parameters
comm   = MPI.COMM_WORLD
size   = comm.Get_size()
rank   = comm.Get_rank()
left   = rank - 1
right  = rank + 1
last   = size - 1
master = rank == 0

# Work size and global indices
work_size_total = None
work_size       = None
work_start      = None
work_end        = None


def recv(source):
    return comm.recv(source=source)

def isend(obj, dest):
    return comm.isend(obj, dest)

def bcast(obj, source):
    return comm.bcast(obj, source)

def exscan(var, buff):
    comm.Exscan(var, buff, MPI.SUM)

def reduce_master(var, buff):
    comm.Reduce(var, buff, MPI.SUM, 0)

def allreduce(var, buff):
    comm.Allreduce(var, buff, MPI.SUM)

def global_idx(N, return_total=False):
    buff = np.array([0], dtype=int)
    exscan(np.array(N, dtype=int), buff)
    start = buff[0]
    end   = buff[0] + N

    if not return_total:
        return start, end
    else:
        N_total = bcast(np.array([end], dtype=int), last)
        return start, end, N_total[0]

def global_wgt(w, total_only=True):
    buff = np.array([0], dtype=float)
    exscan(np.array(w, dtype=float), buff)
    start = buff[0]
    end   = buff[0] + w

    w_total = bcast(np.array([end], dtype=float), last)

    if total_only:
        return w_total
    else:
        return start, end, w_total

def distribute_work(N_work):
    global work_size_total, work_size, work_start, work_end

    # Total # of work
    work_size_total = N_work

    # Local # of work
    work_size = floor(N_work/size)
    if rank < N_work%size:
        work_size += 1

    # Starting and ending work indices
    work_start, work_end = global_idx(work_size)
    

def bank_passing(bank, i_start, i_end):
    # Need more or less?
    more_left  = i_start < work_start
    less_left  = i_start > work_start
    more_right = i_end   > work_end
    less_right = i_end   < work_end

    # Offside?
    offside_left  = i_end   <= work_start
    offside_right = i_start >= work_end

    # If offside, need to receive first
    if offside_left:
        # Receive from right
        bank.extend(recv(right))
        less_right = False
    if offside_right:
        # Receive from left
        bank[0:0] = recv(left)
        less_left = False

    # Send
    if more_left:
        n = work_start - i_start
        request_left = isend(bank[:n], left)
        bank = bank[n:]
    if more_right:
        n = i_end - work_end
        request_right = isend(bank[-n:], right)
        bank = bank[:-n]

    # Receive
    if less_left:
        bank[0:0] = recv(left)
    if less_right: 
        bank.extend(recv(right))

    # Wait unti sent massage is received
    if more_left : request_left.Wait()
    if more_right: request_right.Wait()
    
    return bank

