from   math   import floor
from   mpi4py import MPI
import numpy  as     np

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
work_end        = None # work_start + work_size

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

# Collective
def bcast(buff, root):
    return comm.Bcast(buff, root)
def exscan(var, buff):
    comm.Exscan(var, buff, MPI.SUM)
def reduce_master(var):
    var = np.array(var) # Anticipating Python-converted Numba types
    buff = np.zeros_like(var)
    comm.Reduce(var, buff, MPI.SUM, 0)
    if buff.shape == ():
        return buff
    else:
        return buff[:]
def allreduce(var):
    var = np.array(var) # Anticipating Python-converted Numba types
    buff = np.zeros_like(var)
    comm.Allreduce(var, buff, MPI.SUM)
    if buff.shape == ():
        return buff
    else:
        return buff[:]


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

def distribute_work(N):
    global work_size_total, work_size, work_start, work_end

    # Total # of work
    work_size_total = N

    # Evenly distribute work
    work_size = floor(N/size)

    # Starting index (based on even distribution)
    work_start = work_size*rank

    # Count reminder
    rem = N%size

    # Assign reminder and update starting index
    if rank < rem:
        work_size  += 1
        work_start += rank
    else:
        work_start += rem

    # Ending work index
    work_end = work_start + work_size


# =============================================================================
# Particle bank operations
# =============================================================================

def total_weight(bank):
    W_local = np.zeros(1)
    for i in range(bank['size']):
        W_local[0] += bank['particles'][i]['weight']
    return allreduce(W_local)[0]

def normalize_weight(bank, norm):
    # Get total weight
    W = total_weight(bank)

    # Normalize weight
    for P in bank['particles']:
        P['weight'] *= norm/W

def bank_scanning(bank):
    N_local = bank['size']

    # Starting index
    buff = np.array([0], dtype=int)
    exscan(np.array(N_local, dtype=int), buff)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    bcast(buff, root=last)
    N_global = buff[0]

    return idx_start, N_local, N_global

def bank_passing(bank):
    # Scan the bank
    idx_start, N_local, N = bank_scanning(bank)
    idx_end = idx_start + N_local

    distribute_work(N)

    # Need more or less?
    more_left  = idx_start < work_start
    less_left  = idx_start > work_start
    more_right = idx_end   > work_end
    less_right = idx_end   < work_end

    # Offside?
    offside_left  = idx_end   <= work_start
    offside_right = idx_start >= work_end

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
        n = work_start - idx_start
        request_left = isend(bank[:n], left)
        bank = bank[n:]
    if more_right:
        n = idx_end - work_end
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
