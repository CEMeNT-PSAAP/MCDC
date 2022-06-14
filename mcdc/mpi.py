from   math   import floor
from   mpi4py import MPI
import numpy  as     np

# Communication parameters
comm   = MPI.COMM_WORLD
size   = comm.Get_size()
rank   = comm.Get_rank()
master = rank == 0

# Timer
Wtime = MPI.Wtime

# Barrier
Barrier = comm.Barrier


# =============================================================================
# Basic communications
# =============================================================================

# Point-to-point
def recv(source):
    return comm.recv(source=source)
def isend(obj, dest):
    return comm.isend(obj, dest)

# =============================================================================
# Indexers
# =============================================================================

def global_idx(N):
    buff = np.array([0], dtype=int)
    exscan(np.array(N, dtype=int), buff)
    start = buff[0]
    end   = buff[0] + N
    return start, end

def global_weight(W):
    buff = np.array([0], dtype=float)
    exscan(np.array(W, dtype=float), buff)
    start = buff[0]
    end   = buff[0] + W
    return start, end

'''
def statistics(mean, sdev, total, total_sq, N):
    size  = total.size
    shape = total.shape

    # Distribute index
    idx_start, idx_size, idx_end = work_idx(size)

    # Calculate mean and sdev
    for i in range(idx_start, idx_end):
        idx = np.unravel_index(i, shape)
        mean[idx] = total[idx]/N
        sdev[idx] = np.sqrt((total_sq[idx]/N - np.square(mean[idx]))/(N-1))

def save_hdf5_parallel(file_, name, data):
    size  = data.size
    shape = data.shape

    # Create dataset
    dset = file_.create_dataset(name, shape)

    # Distribute index
    idx_start, idx_size, idx_end = work_idx(size)

    # Store in HDF5
    for i in range(idx_start, idx_end):
        idx = np.unravel_index(i, shape)
        dset[idx] = data[idx]
'''
