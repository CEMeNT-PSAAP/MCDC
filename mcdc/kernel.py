import math

from mpi4py import MPI
from numba  import njit, objmode, literal_unroll

import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.print_   import print_error
from mcdc.type_    import score_list

#==============================================================================
# Random sampling
#==============================================================================

@njit
def sample_isotropic_direction(mcdc):
    # Sample polar cosine and azimuthal angle uniformly
    mu  = 2.0*rng(mcdc) - 1.0
    azi = 2.0*PI*rng(mcdc)

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2)**0.5
    y = math.cos(azi)*c
    z = math.sin(azi)*c
    x = mu
    return x, y, z

@njit
def sample_white_direction(nx, ny, nz, mcdc):
    # Sample polar cosine
    mu = math.sqrt(rng(mcdc))
    
    # Sample azimuthal direction
    azi     = 2.0*PI*rng(mcdc)
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac      = (1.0 - mu**2)**0.5
    
    if nz != 1.0:
        B = (1.0 - nz**2)**0.5
        C = Ac/B
        
        x = nx*mu + (nx*nz*cos_azi - ny*sin_azi)*C
        y = ny*mu + (ny*nz*cos_azi + nx*sin_azi)*C
        z = nz*mu - cos_azi*Ac*B
    
    # If dir = 0i + 0j + k, interchange z and y in the formula
    else:
        B = (1.0 - ny**2)**0.5
        C = Ac/B
        
        x = nx*mu + (nx*ny*cos_azi - nz*sin_azi)*C
        y = nz*mu + (nz*ny*cos_azi + nx*sin_azi)*C
        z = ny*mu - cos_azi*Ac*B
    return x, y, z

@njit
def sample_uniform(a, b, mcdc):
    return a + rng(mcdc) * (b - a)

# TODO: use cummulative density function and binary search
@njit
def sample_discrete(p, mcdc):
    tot = 0.0
    xi  = rng(mcdc)
    for i in range(p.shape[0]):
        tot += p[i]
        if tot > xi:
            return i

#==============================================================================
# Random number generator operations
#==============================================================================
# TODO: make g, c, and mod constants

@njit
def rng_rebase(mcdc):
    mcdc['rng_seed_base'] = mcdc['rng_seed']

@njit
def rng_skip_ahead_strides(n, mcdc):
    rng_skip_ahead_(int(n*mcdc['rng_stride']), mcdc)

@njit
def rng_skip_ahead(n, mcdc):
    rng_skip_ahead_(int(n), mcdc)


@njit
def rng_skip_ahead_(n, mcdc):
    seed_base = mcdc['rng_seed_base']
    g         = int(mcdc['setting']['rng_g'])
    c         = int(mcdc['setting']['rng_c'])
    g_new     = 1
    c_new     = 0
    mod       = int(mcdc['setting']['rng_mod'])
    mod_mask  = int(mod - 1)
    
    n = n & mod_mask
    while n > 0:
        if n & 1:
            g_new = g_new*g       & mod_mask
            c_new = (c_new*g + c) & mod_mask

        c = (g+1)*c & mod_mask
        g = g*g     & mod_mask
        n >>= 1
    
    mcdc['rng_seed'] = (g_new*int(seed_base) + c_new ) & mod_mask

@njit
def rng(mcdc):
    seed     = int(mcdc['rng_seed'])
    g        = int(mcdc['setting']['rng_g'])
    c        = int(mcdc['setting']['rng_c'])
    mod      = int(mcdc['setting']['rng_mod'])
    mod_mask = int(mod - 1)

    mcdc['rng_seed'] = (g*int(seed) + c) & mod_mask
    return mcdc['rng_seed']/mod

#==============================================================================
# Particle source operations
#==============================================================================

@njit
def source_particle(source, rng):
    # Position
    if source['box']:
        x = sample_uniform(source['box_x'][0], source['box_x'][1], rng)
        y = sample_uniform(source['box_y'][0], source['box_y'][1], rng)
        z = sample_uniform(source['box_z'][0], source['box_z'][1], rng)
    else:
        x = source['x']
        y = source['y']
        z = source['z']

    # Direction
    if source['isotropic']:
        ux, uy, uz = sample_isotropic_direction(rng)
    elif source['white']:
        ux, uy, uz = sample_white_direction(source['white_x'], 
                                            source['white_y'], 
                                            source['white_z'], rng)
    else:
        ux = source['ux']
        uy = source['uy']
        uz = source['uz']

    # Energy and time
    g = sample_discrete(source['group'], rng)
    t = sample_uniform(source['time'][0], source['time'][1], rng)

    # Make and return particle
    P       = np.zeros(1, dtype=type_.particle_record)[0]
    P['x']  = x
    P['y']  = y
    P['z']  = z
    P['t']  = t
    P['ux'] = ux
    P['uy'] = uy
    P['uz'] = uz
    P['g']  = g
    P['w']  = 1.0

    return P

#==============================================================================
# Particle bank operations
#==============================================================================

@njit
def add_particle(P, bank):
    # Check if bank is full
    if bank['size'] == bank['particles'].shape[0]:
        with objmode():
            print_error('Particle %s bank is full.'%bank['tag'])

    # Set particle
    bank['particles'][bank['size']] = P

    # Increment size
    bank['size'] += 1

@njit
def get_particle(bank):
    # Check if bank is empty
    if bank['size'] == 0:
        with objmode():
            print_error('Particle %s bank is empty.'%bank['tag'])

    # Decrement size
    bank['size'] -= 1

    # Create in-flight particle
    P = np.zeros(1, dtype=type_.particle)[0]

    # Set attribute
    P_rec = bank['particles'][bank['size']]
    P['x']  = P_rec['x']
    P['y']  = P_rec['y']
    P['z']  = P_rec['z']
    P['t']  = P_rec['t']
    P['ux'] = P_rec['ux']
    P['uy'] = P_rec['uy']
    P['uz'] = P_rec['uz']
    P['g']  = P_rec['g']
    P['w']  = P_rec['w']
    P['alive'] = True

    # Set default IDs and event
    P['material_ID'] = -1
    P['cell_ID']     = -1
    P['surface_ID']  = -1
    P['event']       = -1
    return P

@njit
def manage_particle_banks(mcdc):
    # Record time
    if mcdc['mpi_master']:
        with objmode(time_start='float64'):
            time_start = MPI.Wtime()

    if mcdc['setting']['mode_eigenvalue']:
        # Normalize weight
        normalize_weight(mcdc['bank_census'], mcdc['setting']['N_particle'])

    # Sync RNG
    skip = mcdc['mpi_work_size_total']-mcdc['mpi_work_start']
    rng_skip_ahead_strides(skip, mcdc)
    rng_rebase(mcdc)

    # Population control
    if mcdc['technique']['population_control']:
        population_control(mcdc)
    else:
        # TODO: Swap??
        # Swap census and source bank
        for i in range(mcdc['bank_census']['size']):
            mcdc['bank_source']['particles'][i] = \
                    copy_particle(mcdc['bank_census']['particles'][i])
        mcdc['bank_source']['size'] = mcdc['bank_census']['size']

    # MPI rebalance
    bank_rebalance(mcdc)
    
    # Zero out census bank
    mcdc['bank_census']['size'] = 0

    # Manage IC bank
    if mcdc['technique']['IC_generator'] and mcdc['cycle_active']:
        manage_IC_bank(mcdc)
    
    # Accumulate time
    if mcdc['mpi_master']:
        with objmode(time_end='float64'):
            time_end = MPI.Wtime()
        mcdc['runtime_bank_management'] += time_end - time_start

@njit
def manage_IC_bank(mcdc):
    # Buffer bank
    buff_n = np.zeros(mcdc['technique']['IC_bank_neutron_local']['content'].shape[0], 
                      dtype=type_.neutron)
    buff_p = np.zeros(mcdc['technique']['IC_bank_precursor_local']['content'].shape[0], 
                      dtype=type_.precursor)

    # Resample?
    if mcdc['technique']['IC_resample']:
        size_n = mcdc['technique']['IC_bank_neutron_local']['size']
        size_C = mcdc['technique']['IC_bank_precursor_local']['size']
        Pmax_n = mcdc['technique']['IC_Pmax_n']
        Pmax_C = mcdc['technique']['IC_Pmax_C']

        print(size_n, size_C)

        # Neutron
        Nn = 0
        # Sample and store to buffer
        for i in range(size_n):
            P = mcdc['technique']['IC_bank_neutron_local']['content'][i]
            if rng(mcdc) < Pmax_n*P['w']:
                P['w']      = 1.0/Pmax_n
                buff_n[Nn]  = P
                Nn         += 1
        # Set actual bank
        for i in range(Nn):
            mcdc['technique']['IC_bank_neutron_local']['content'][i] = buff_n[i]
        mcdc['technique']['IC_bank_neutron_local']['size'] = Nn
        
        # Precursor
        Np = 0
        # Sample and store to buffer
        for i in range(size_C):
            P = mcdc['technique']['IC_bank_precursor_local']['content'][i]
            if rng(mcdc) < Pmax_C*P['w']:
                P['w']      = 1.0/Pmax_C
                buff_p[Np]  = P
                Np         += 1
        # Set actual bank
        for i in range(Np):
            mcdc['technique']['IC_bank_precursor_local']['content'][i] = buff_p[i]
        mcdc['technique']['IC_bank_precursor_local']['size'] = Np

        # Reset parameters
        mcdc['technique']['IC_Pmax_n'] = 0.0
        mcdc['technique']['IC_Pmax_C'] = 0.0
        
        print(Nn, Np)

    with objmode(Nn='int64', Np='int64'):
        # Create MPI-supported numpy object
        Nn = mcdc['technique']['IC_bank_neutron_local']['size']
        Np = mcdc['technique']['IC_bank_precursor_local']['size']
    
        neutrons   = MPI.COMM_WORLD.gather(
                mcdc['technique']['IC_bank_neutron_local']['content'][:Nn])
        precursors = MPI.COMM_WORLD.gather(
                mcdc['technique']['IC_bank_precursor_local']['content'][:Np])

        if mcdc['mpi_master']:
            neutrons   = np.concatenate(neutrons[:])
            precursors = np.concatenate(precursors[:])

            # Set output buffer
            Nn = neutrons.shape[0]
            Np = precursors.shape[0]
            for i in range(Nn):
                buff_n[i] = neutrons[i]
            for i in range(Np):
                buff_p[i] = precursors[i]

    # Set global bank from buffer
    if mcdc['mpi_master']:
        start_n = mcdc['technique']['IC_bank_neutron']['size']
        start_p = mcdc['technique']['IC_bank_precursor']['size']
        mcdc['technique']['IC_bank_neutron']['size']   += Nn
        mcdc['technique']['IC_bank_precursor']['size'] += Np
        for i in range(Nn):
            mcdc['technique']['IC_bank_neutron']['content'][start_n+i] = buff_n[i]
        for i in range(Np):
            mcdc['technique']['IC_bank_precursor']['content'][start_p+i] = buff_p[i]

    # Reset local banks
    mcdc['technique']['IC_bank_neutron_local']['size'] = 0
    mcdc['technique']['IC_bank_precursor_local']['size'] = 0

@njit
def bank_scanning(bank, mcdc):
    N_local = bank['size']

    # Starting index
    buff = np.zeros(1, dtype=np.int64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([N_local]), buff, MPI.SUM)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc['mpi_size']-1)
    N_global = buff[0]

    return idx_start, N_local, N_global

@njit
def bank_scanning_weight(bank, mcdc):
    # Local weight CDF
    N_local = bank['size']
    w_cdf   = np.zeros(N_local+1)
    for i in range(N_local):
        w_cdf[i+1] = w_cdf[i] + bank['particles'][i]['w']
    W_local = w_cdf[-1]

    # Starting weight
    buff = np.zeros(1, dtype=np.float64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([W_local]), buff, MPI.SUM)
    w_start = buff[0]
    w_cdf += w_start

    # Global weight
    buff[0] = w_cdf[-1]
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc['mpi_size']-1)
    W_global = buff[0]

    return w_start, w_cdf, W_global

@njit
def normalize_weight(bank, norm):
    # Get total weight
    W = total_weight(bank)

    # Normalize weight
    for P in bank['particles']:
        P['w'] *= norm/W

@njit
def total_weight(bank):
    # Local total weight
    W_local = np.zeros(1)
    for i in range(bank['size']):
        W_local[0] += bank['particles'][i]['w']
    
    # MPI Allreduce
    buff = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(W_local, buff, MPI.SUM)
    return buff[0]

@njit
def bank_rebalance(mcdc):
    # Scan the bank
    idx_start, N_local, N = bank_scanning(mcdc['bank_source'], mcdc)
    idx_end = idx_start + N_local

    distribute_work(N, mcdc)

    # Some constants
    work_start = mcdc['mpi_work_start']
    work_end   = work_start + mcdc['mpi_work_size']
    left       = mcdc['mpi_rank'] - 1
    right      = mcdc['mpi_rank'] + 1

    # Need more or less?
    more_left  = idx_start < work_start
    less_left  = idx_start > work_start
    more_right = idx_end   > work_end
    less_right = idx_end   < work_end

    # Offside?
    offside_left  = idx_end   <= work_start
    offside_right = idx_start >= work_end

    # MPI nearest-neighbor send/receive
    buff = np.zeros(mcdc['bank_source']['particles'].shape[0], 
                    dtype=type_.particle_record)

    with objmode(size='int64'):
        # Create MPI-supported numpy object
        size = mcdc['bank_source']['size']
        bank = np.array(mcdc['bank_source']['particles'][:size])

        # If offside, need to receive first
        if offside_left:
            # Receive from right
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))
            less_right = False
        if offside_right:
            # Receive from left
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
            less_left = False

        # Send
        if more_left:
            n = work_start - idx_start
            request_left = MPI.COMM_WORLD.isend(bank[:n], dest=left)
            bank = bank[n:]
        if more_right:
            n = idx_end - work_end
            request_right = MPI.COMM_WORLD.isend(bank[-n:], dest=right)
            bank = bank[:-n]

        # Receive
        if less_left:
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
        if less_right: 
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))

        # Wait until sent massage is received
        if more_left : request_left.Wait()
        if more_right: request_right.Wait()

        # Set output buffer
        size = bank.shape[0]
        for i in range(size):
            buff[i] = bank[i]
        
    # Set source bank from buffer
    mcdc['bank_source']['size'] = size
    for i in range(size):
        mcdc['bank_source']['particles'][i] = buff[i]
        
@njit
def distribute_work(N, mcdc):
    size = mcdc['mpi_size']
    rank = mcdc['mpi_rank']

    # Total # of work
    work_size_total = N

    # Evenly distribute work
    work_size = math.floor(N/size)

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

    mcdc['mpi_work_start']      = work_start
    mcdc['mpi_work_size']       = work_size
    mcdc['mpi_work_size_total'] = work_size_total

# =============================================================================
# IC generator
# =============================================================================

@njit
def bank_IC(P, mcdc):
    material = mcdc['materials'][P['material_ID']] 

    #==========================================================================
    # Neutron
    #==========================================================================

    # Neutron weight
    g      = P['g']
    SigmaT = material['total'][g]
    weight = P['w']
    flux   = weight/SigmaT
    v      = material['speed'][g]
    wn     = flux/v

    # Neutron target weight
    Nn       = mcdc['technique']['IC_N_neutron']
    tally_n  = mcdc['technique']['IC_n_eff']
    N_cycle  = mcdc['setting']['N_active']
    wn_prime = tally_n*N_cycle/Nn
    
    # Sampling probability
    Pn = wn/wn_prime
    
    # Sample neutron
    if Pn > 1.0:
        wn_prime = wn
        if Pn > mcdc['technique']['IC_Pmax_n']:
            mcdc['technique']['IC_Pmax_n'] = Pn

    if rng(mcdc) < Pn:
        idx     = mcdc['technique']['IC_bank_neutron_local']['size']
        neutron = mcdc['technique']['IC_bank_neutron_local']['content'][idx]
        neutron['x']  = P['x']
        neutron['y']  = P['y']
        neutron['z']  = P['z']
        neutron['ux'] = P['ux']
        neutron['uy'] = P['uy']
        neutron['uz'] = P['uz']
        neutron['g']  = P['g']
        neutron['w']  = wn_prime
        mcdc['technique']['IC_bank_neutron_local']['size'] += 1
    
    #==========================================================================
    # Precursor
    #==========================================================================

    # Sample precursor?
    Np = mcdc['technique']['IC_N_precursor']
    if Np == 0:
        return

    # Precursor weight
    J      = material['J']
    nu_d   = material['nu_d'][g]
    SigmaF = material['fission'][g]
    decay  = material['decay']
    total  = 0.0
    for j in range(J):
        total += nu_d[j]/decay[j]
    wp = flux*total*SigmaF/mcdc['k_eff']

    # Precursor target weight
    tally_C  = mcdc['technique']['IC_C_eff']
    wp_prime = tally_C*N_cycle/Np

    # Sampling probability
    Pp = wp/wp_prime

    # Sample precursor
    if Pp > 1.0:
        wp_prime = wp
        if Pp > mcdc['technique']['IC_Pmax_C']:
            mcdc['technique']['IC_Pmax_C'] = Pp

    if rng(mcdc) < Pp:
        idx       = mcdc['technique']['IC_bank_precursor_local']['size']
        precursor = mcdc['technique']['IC_bank_precursor_local']['content'][idx]
        precursor['x'] = P['x']
        precursor['y'] = P['y']
        precursor['z'] = P['z']
        precursor['w'] = wp_prime
        mcdc['technique']['IC_bank_precursor_local']['size'] += 1

        # Sample group
        xi    = rng(mcdc)*total
        total = 0.0
        for j in range(J):
            total += nu_d[j]/decay[j]
            if total > xi:
                break
        precursor['g'] = j

# =============================================================================
# Population control techniques
# =============================================================================
# TODO: Make it a stand-alone function that takes (bank_init, bank_final, M).
#       The challenge is in the use of type-dependent copy_particle which is 
#       required due to pure-Python behavior of taking things by reference.

@njit
def population_control(mcdc):
    if mcdc['technique']['pct'] == PCT_COMBING:
        pct_combing(mcdc)
        rng_rebase(mcdc)
    elif mcdc['technique']['pct'] == PCT_COMBING_WEIGHT:
        pct_combing_weight(mcdc)
        rng_rebase(mcdc)

@njit
def pct_combing(mcdc):
    bank_census = mcdc['bank_census']
    M           = mcdc['setting']['N_particle']
    bank_source = mcdc['bank_source']
    
    # Scan the bank
    idx_start, N_local, N = bank_scanning(bank_census, mcdc)
    idx_end = idx_start + N_local

    # Teeth distance
    td = N/M

    # Tooth offset
    xi     = rng(mcdc)
    offset = xi*td

    # First hiting tooth
    tooth_start = math.ceil((idx_start-offset)/td)

    # Last hiting tooth
    tooth_end = math.floor((idx_end-offset)/td) + 1

    # Locally sample particles from census bank
    bank_source['size'] = 0
    for i in range(tooth_start, tooth_end):
        tooth = i*td+offset
        idx   = math.floor(tooth) - idx_start
        P = copy_particle(bank_census['particles'][idx])
        # Set weight
        P['w'] *= td
        add_particle(P, bank_source)

@njit
def pct_combing_weight(mcdc):
    bank_census = mcdc['bank_census']
    M           = mcdc['setting']['N_particle']
    bank_source = mcdc['bank_source']
    
    # Scan the bank based on weight
    w_start, w_cdf, W = bank_scanning_weight(bank_census, mcdc)
    w_end = w_cdf[-1]

    # Teeth distance
    td = W/M

    # Tooth offset
    xi     = rng(mcdc)
    offset = xi*td

    # First hiting tooth
    tooth_start = math.ceil((w_start-offset)/td)

    # Last hiting tooth
    tooth_end = math.floor((w_end-offset)/td) + 1

    # Locally sample particles from census bank
    bank_source['size'] = 0
    idx = 0
    for i in range(tooth_start, tooth_end):
        tooth  = i*td+offset
        idx   += binary_search(tooth,w_cdf[idx:])
        P = copy_particle(bank_census['particles'][idx])
        # Set weight
        P['w'] = td
        add_particle(P, bank_source)
'''
@njit
def pct_IC_neutron(mcdc):
    bank_census = mcdc['bank_census']
    M           = mcdc['setting']['N_particle']
    bank_source = mcdc['bank_source']
    
    # Scan the bank based on weight
    w_start, w_cdf, W = bank_scanning_weight(bank_census, mcdc)
    w_end = w_cdf[-1]

    # Teeth distance
    td = W/M

    # Tooth offset
    xi     = rng(mcdc)
    offset = xi*td

    # First hiting tooth
    tooth_start = math.ceil((w_start-offset)/td)

    # Last hiting tooth
    tooth_end = math.floor((w_end-offset)/td) + 1

    # Locally sample particles from census bank
    bank_source['size'] = 0
    idx = 0
    for i in range(tooth_start, tooth_end):
        tooth  = i*td+offset
        idx   += binary_search(tooth,w_cdf[idx:])
        P = copy_particle(bank_census['particles'][idx])
        # Set weight
        P['w'] = td
        add_particle(P, bank_source)
'''
#==============================================================================
# Particle operations
#==============================================================================

@njit
def move_particle(P, distance, mcdc):
    P['x'] += P['ux']*distance
    P['y'] += P['uy']*distance
    P['z'] += P['uz']*distance
    P['t'] += distance/get_particle_speed(P, mcdc)

@njit
def shift_particle(P, shift):
    if P['ux'] > 0.0:
        P['x'] += shift
    else:
        P['x'] -= shift
    if P['uy'] > 0.0:
        P['y'] += shift
    else:
        P['y'] -= shift
    if P['uz'] > 0.0:
        P['z'] += shift
    else:
        P['z'] -= shift
    P['t'] += shift

@njit
def get_particle_cell(P, universe_ID, trans, mcdc):
    """
    Find and return particle cell ID in the given universe and translation
    """

    universe = mcdc['universes'][universe_ID]
    for cell_ID in universe['cell_IDs']:
        cell = mcdc['cells'][cell_ID]
        if cell_check(P, cell, trans, mcdc):
            return cell['ID']

    # Particle is not found
    print("A particle is lost at (",P['x'],P['y'],P['z'],")")
    P['alive'] = False
    return -1

@njit
def get_particle_speed(P, mcdc):
    return mcdc['materials'][P['material_ID']]['speed'][P['g']]

@njit
def copy_particle(P):
    P_new = np.zeros(1, dtype=type_.particle_record)[0]
    P_new['x']  = P['x']         
    P_new['y']  = P['y']         
    P_new['z']  = P['z']         
    P_new['t']  = P['t']         
    P_new['ux'] = P['ux']        
    P_new['uy'] = P['uy']        
    P_new['uz'] = P['uz']        
    P_new['g']  = P['g']     
    P_new['w']  = P['w']     
    return P_new

#==============================================================================
# Cell operations
#==============================================================================

@njit
def cell_check(P, cell, trans, mcdc):
    for i in range(cell['N_surface']):
        surface = mcdc['surfaces'][cell['surface_IDs'][i]]
        result  = surface_evaluate(P, surface, trans)
        if cell['positive_flags'][i]:
            if result < 0.0: return False
        else:
            if result > 0.0: return False
    return True

#==============================================================================
# Surface operations
#==============================================================================
# Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J(t) = 0
#   J(t) = J0_i + J1_i*t for t in [t_{i-1}, t_i), t_0 = 0

@njit
def surface_evaluate(P, surface, trans):
    x = P['x'] + trans[0]
    y = P['y'] + trans[1]
    z = P['z'] + trans[2]
    t = P['t']
    
    G = surface['G']
    H = surface['H']
    I = surface['I']

    # Get time indices
    idx = 0
    if surface['N_slice'] > 1:
        idx = binary_search(t, surface['t'][:surface['N_slice']+1])

    # Get constant
    J0 = surface['J'][idx][0]
    J1 = surface['J'][idx][1]
    J = J0 + J1*(t-surface['t'][idx])

    result = G*x + H*y + I*z + J
    
    if surface['linear']:
        return result

    A = surface['A']
    B = surface['B']
    C = surface['C']
    D = surface['D']
    E = surface['E']
    F = surface['F']
    
    return result + A*x*x + B*y*y + C*z*z + D*x*y + E*x*z + F*y*z              

@njit
def surface_bc(P, surface, trans):
    if surface['vacuum']:
        P['alive'] = False
    elif surface['reflective']:
        surface_reflect(P, surface, trans)

@njit
def surface_reflect(P, surface, trans):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']
    nx, ny, nz  = surface_normal(P, surface, trans)
    c  = 2.0*(nx*ux + ny*uy + nz*uz) # 2.0*surface_normal_component(...)

    P['ux'] = ux - c*nx
    P['uy'] = uy - c*ny
    P['uz'] = uz - c*nz

@njit
def surface_shift(P, surface, trans, mcdc):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']

    # Get surface normal
    nx, ny, nz = surface_normal(P, surface, trans)

    # The shift
    shift_x = nx*SHIFT
    shift_y = ny*SHIFT
    shift_z = nz*SHIFT

    # Get dot product to determine shift sign
    if surface['linear']:
        # Get time indices
        idx = 0
        if surface['N_slice'] > 1:
            idx = binary_search(P['t'], surface['t'][:surface['N_slice']+1])
        J1  = surface['J'][idx][1]
        v   = get_particle_speed(P, mcdc)
        dot = ux*nx + uy*ny + uz*nz + J1/v
    else:
        dot = ux*nx + uy*ny + uz*nz

    if dot > 0.0:
        P['x'] += shift_x
        P['y'] += shift_y
        P['z'] += shift_z
    else:
        P['x'] -= shift_x
        P['y'] -= shift_y
        P['z'] -= shift_z

@njit
def surface_normal(P, surface, trans):
    if surface['linear']:
        return surface['nx'], surface['ny'], surface['nz']
    
    A = surface['A']
    B = surface['B']
    C = surface['C']
    D = surface['D']
    E = surface['E']
    F = surface['F']
    G = surface['G']
    H = surface['H']
    I = surface['I']
    x = P['x'] + trans[0]
    y = P['y'] + trans[1]
    z = P['z'] + trans[2]
    
    dx = 2*A*x + D*y + E*z + G
    dy = 2*B*y + D*x + F*z + H
    dz = 2*C*z + E*x + F*y + I
    
    norm = (dx**2 + dy**2 + dz**2)**0.5
    return dx/norm, dy/norm, dz/norm
    
@njit
def surface_normal_component(P, surface, trans):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']
    nx, ny, nz  = surface_normal(P, surface, trans)
    return nx*ux + ny*uy + nz*uz

@njit
def surface_distance(P, surface, trans, mcdc):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']

    G  = surface['G']
    H  = surface['H']
    I  = surface['I']
    
    surface_move = False
    if surface['linear']:
        idx = 0
        if surface['N_slice'] > 1:
            idx = binary_search(P['t'], surface['t'][:surface['N_slice']+1])
        J1  = surface['J'][idx][1]
        v   = get_particle_speed(P, mcdc)

        t_max = surface['t'][idx+1]
        d_max = (t_max - P['t'])*v

        distance = -surface_evaluate(P, surface, trans)\
                    /(G*ux + H*uy + I*uz + J1/v)
        
        # Go beyond current movement slice?
        if distance > d_max:
            distance = d_max
            surface_move = True
        elif distance < 0 and idx < surface['N_slice']-1:
            distance = d_max
            surface_move = True

        # Moving away from the surface
        if distance < 0.0: return INF, surface_move
        else:              return distance, surface_move
        
    x  = P['x'] + trans[0] 
    y  = P['y'] + trans[1] 
    z  = P['z'] + trans[2] 

    A  = surface['A']
    B  = surface['B']
    C  = surface['C']
    D  = surface['D']
    E  = surface['E']
    F  = surface['F']

    # Quadratic equation constants
    a = A*ux*ux + B*uy*uy + C*uz*uz + D*ux*uy + E*ux*uz + F*uy*uz
    b = 2*(A*x*ux + B*y*uy + C*z*uz) +\
        D*(x*uy + y*ux) + E*(x*uz + z*ux) + F*(y*uz + z*uy) +\
        G*ux + H*uy + I*uz
    c = surface_evaluate(P, surface, trans)
    
    determinant = b*b - 4.0*a*c
    
    # Roots are complex  : no intersection
    # Roots are identical: tangent
    # ==> return huge number
    if determinant <= 0.0:
        return INF, surface_move
    else:
        # Get the roots
        denom = 2.0*a
        sqrt  = math.sqrt(determinant)
        root_1 = (-b + sqrt)/denom
        root_2 = (-b - sqrt)/denom
        
        # Negative roots, moving away from the surface
        if root_1 < 0.0: root_1 = INF
        if root_2 < 0.0: root_2 = INF
        
        # Return the smaller root
        return min(root_1, root_2), surface_move

#==============================================================================
# Mesh operations
#==============================================================================

@njit
def mesh_distance_search(value, direction, grid):
    if direction == 0.0:
        return INF
    idx = binary_search(value, grid)
    if direction > 0.0:
        idx += 1
    if idx == -1: idx += 1
    if idx == len(grid): idx -= 1
    dist = (grid[idx] - value)/direction
    # Moving away from mesh?
    if dist < 0.0:
        dist = INF
    return dist

@njit
def mesh_uniform_distance_search(value, direction, x0, dx):
    if direction == 0.0:
        return INF
    idx = math.floor((value - x0)/dx)
    if direction > 0.0:
        idx += 1
    ref = x0 + idx*dx
    dist = (ref - value)/direction
    return dist

@njit
def mesh_get_index(P, mesh):
    # Check if outside grid
    outside = False

    if P['t'] < mesh['t'][0] or P['t'] > mesh['t'][-1] or\
       P['x'] < mesh['x'][0] or P['x'] > mesh['x'][-1] or\
       P['y'] < mesh['y'][0] or P['y'] > mesh['y'][-1] or\
       P['z'] < mesh['z'][0] or P['z'] > mesh['z'][-1]:
        outside = True

    t = binary_search(P['t'], mesh['t'])
    x = binary_search(P['x'], mesh['x'])
    y = binary_search(P['y'], mesh['y'])
    z = binary_search(P['z'], mesh['z'])
    return t, x, y, z, outside

@njit
def mesh_get_angular_index(P, mesh):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']

    P_mu  = uz
    P_azi = math.acos(ux/math.sqrt(ux*ux + uy*uy))
    if uy < 0.0: 
        P_azi *= -1

    mu  = binary_search(P_mu, mesh['mu'])
    azi = binary_search(P_azi, mesh['azi'])
    return mu, azi

@njit
def mesh_uniform_get_index(P, mesh, trans):
    Px = P['x'] + trans[0]
    Py = P['y'] + trans[1]
    Pz = P['z'] + trans[2]
    x  = math.floor((Px - mesh['x0'])/mesh['dx'])
    y  = math.floor((Py - mesh['y0'])/mesh['dy'])
    z  = math.floor((Pz - mesh['z0'])/mesh['dz'])
    return x, y, z

@njit
def mesh_crossing_evaluate(P, mesh):
    # Shift backward
    shift_particle(P, -SHIFT)
    t1, x1, y1, z1, outside = mesh_get_index(P, mesh)
   
    # Double shift forward
    shift_particle(P, 2*SHIFT)
    t2, x2, y2, z2, outside = mesh_get_index(P, mesh)

    # Return particle to initial position
    shift_particle(P, -SHIFT)
    
    # Determine dimension crossed
    if x1 != x2:
        return x1, y1, z1, t1, MESH_X
    elif y1 != y2:
        return x1, y1, z1, t1, MESH_Y
    elif z1 != z2:
        return x1, y1, z1, t1, MESH_Z
    elif t1 != t2:
        return x1, y1, z1, t1, MESH_T

#==============================================================================
# Tally operations
#==============================================================================

@njit
def score_tracklength(P, distance, mcdc):
    tally    = mcdc['tally']
    material = mcdc['materials'][P['material_ID']]

    # Get indices
    g = P['g']
    t, x, y, z, outside = mesh_get_index(P, tally['mesh'])
    mu, azi = mesh_get_angular_index(P, tally['mesh'])

    # Outside grid?
    if outside:
        return

    # Score
    flux = distance*P['w']
    if tally['flux']:
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['flux'])
    if tally['density']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['density'])
    if tally['fission']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['fission'])
    if tally['total']:
        flux *= material['total'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['total'])
    if tally['current']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current'])
    if tally['eddington']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington'])

@njit
def score_crossing_x(P, t, x, y, z, mcdc):
    tally    = mcdc['tally']
    material = mcdc['materials'][P['material_ID']]

    # Get indices
    g = P['g']
    if P['ux'] > 0.0:
        x += 1
    mu, azi = mesh_get_angular_index(P, tally['mesh'])

    # Score
    flux = P['w']/abs(P['ux'])
    if tally['flux_x']:
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['flux_x'])
    if tally['density_x']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['density_x'])
    if tally['fission_x']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['fission_x'])
    if tally['total_x']:
        flux *= material['total'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['total_x'])
    if tally['current_x']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_x'])
    if tally['eddington_x']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_x'])

@njit
def score_crossing_y(P, t, x, y, z, mcdc):
    tally    = mcdc['tally']
    material = mcdc['materials'][P['material_ID']]

    # Get indices
    g = P['g']
    if P['uy'] > 0.0:
        y += 1
    mu, azi = mesh_get_angular_index(P, tally['mesh'])

    # Score
    flux = P['w']/abs(P['uy'])
    if tally['flux_y']:
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['flux_y'])
    if tally['density_y']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['density_y'])
    if tally['fission_y']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['fission_y'])
    if tally['total_y']:
        flux *= material['total'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['total_y'])
    if tally['current_y']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_y'])
    if tally['eddington_y']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_y'])

@njit
def score_crossing_z(P, t, x, y, z, mcdc):
    tally    = mcdc['tally']
    material = mcdc['materials'][P['material_ID']]

    # Get indices
    g = P['g']
    if P['uz'] > 0.0:
        z += 1
    mu, azi = mesh_get_angular_index(P, tally['mesh'])

    # Score
    flux = P['w']/abs(P['uz'])
    if tally['flux_z']:
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['flux_z'])
    if tally['density_z']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['density_z'])
    if tally['fission_z']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['fission_z'])
    if tally['total_z']:
        flux *= material['total'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['total_z'])
    if tally['current_z']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_z'])
    if tally['eddington_z']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_z'])

@njit
def score_crossing_t(P, t, x, y, z, mcdc):
    tally    = mcdc['tally']
    material = mcdc['materials'][P['material_ID']]
    
    # Get indices
    g  = P['g']
    t += 1
    mu, azi = mesh_get_angular_index(P, tally['mesh'])

    # Score
    flux = P['w']*material['speed'][g]
    if tally['flux_t']:
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['flux_t'])
    if tally['density_t']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['density_t'])
    if tally['fission_t']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['fission_t'])
    if tally['total_t']:
        flux *= material['total'][g]
        score_flux(g, t, x, y, z, mu, azi, flux, tally['score']['total_t'])
    if tally['current_t']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_t'])
    if tally['eddington_t']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_t'])

@njit
def score_flux(g, t, x, y, z, mu, azi, flux, score):
    score['bin'][g, t, x, y, z, mu, azi] += flux

@njit
def score_current(g, t, x, y, z, flux, P, score):
    score['bin'][g, t, x, y, z, 0] += flux*P['ux']
    score['bin'][g, t, x, y, z, 1] += flux*P['uy']
    score['bin'][g, t, x, y, z, 2] += flux*P['uz']

@njit
def score_eddington(g, t, x, y, z, flux, P, score):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']
    score['bin'][g, t, x, y, z, 0] += flux*ux*ux
    score['bin'][g, t, x, y, z, 1] += flux*ux*uy
    score['bin'][g, t, x, y, z, 2] += flux*ux*uz
    score['bin'][g, t, x, y, z, 3] += flux*uy*uy
    score['bin'][g, t, x, y, z, 4] += flux*uy*uz
    score['bin'][g, t, x, y, z, 5] += flux*uz*uz

@njit
def score_closeout_history(score, mcdc):
    # Normalize if eigenvalue mode
    if mcdc['setting']['mode_eigenvalue']:
        score['bin'][:] /= mcdc['setting']['N_particle']
        
        # MPI Reduce
        buff = np.zeros_like(score['bin'])
        with objmode():
            MPI.COMM_WORLD.Reduce(np.array(score['bin']), buff, MPI.SUM, 0)
        score['bin'][:] = buff

    # Accumulate score and square of score into mean and sdev
    score['mean'][:] += score['bin']
    score['sdev'][:] += np.square(score['bin'])

    # Reset bin
    score['bin'].fill(0.0)

@njit
def score_closeout(score, mcdc):
    N_history = mcdc['setting']['N_particle']

    if mcdc['setting']['mode_eigenvalue']:
        N_history = mcdc['setting']['N_active']
    else:
        # MPI Reduce
        buff    = np.zeros_like(score['mean'])
        buff_sq = np.zeros_like(score['sdev'])
        with objmode():
            MPI.COMM_WORLD.Reduce(np.array(score['mean']), buff, MPI.SUM, 0)
            MPI.COMM_WORLD.Reduce(np.array(score['sdev']), buff_sq, MPI.SUM, 0)
        score['mean'][:] = buff
        score['sdev'][:] = buff_sq
   
    # Store results
    score['mean'][:] = score['mean']/N_history
    score['sdev'][:] = \
            np.sqrt((score['sdev']/N_history - np.square(score['mean']))\
            /(N_history-1))
    

@njit
def tally_closeout_history(mcdc):
    tally = mcdc['tally']

    for name in literal_unroll(score_list):
        if tally[name]:
            score_closeout_history(tally['score'][name], mcdc)

@njit
def tally_closeout(mcdc):
    tally = mcdc['tally']

    for name in literal_unroll(score_list):
        if tally[name]:
            score_closeout(tally['score'][name], mcdc)

#==============================================================================
# Global tally operations
#==============================================================================

@njit
def global_tally(P, distance, mcdc):
    tally    = mcdc['tally']
    material = mcdc['materials'][P['material_ID']]

    # Parameters
    flux     = distance*P['w']
    g        = P['g']
    nu       = material['nu_p'][g] + sum(material['nu_d'][g])
    SigmaF   = material['fission'][g]
    nuSigmaF = nu*SigmaF

    mcdc['global_tally_nuSigmaF'] += flux*nuSigmaF

    # IC generator tally
    if mcdc['technique']['IC_generator']:
        # Neutron
        v = get_particle_speed(P, mcdc)
        mcdc['technique']['IC_tally_n'] += flux/v

        # Precursor
        J     = material['J']
        nu_d  = material['nu_d'][g]
        decay = material['decay']
        total = 0.0
        for j in range(J):
            total += nu_d[j]/decay[j]
        mcdc['technique']['IC_tally_C'] += flux*total*SigmaF/mcdc['k_eff']

@njit
def global_tally_closeout_history(mcdc):
    N_particle = mcdc['setting']['N_particle']

    i_cycle = mcdc['i_cycle']

    # MPI Allreduce
    buff_nuSigmaF = np.zeros(1, np.float64)
    buff_IC_n     = np.zeros(1, np.float64)
    buff_IC_C     = np.zeros(1, np.float64)
    buff_Pmax_n   = np.zeros(1, np.float64)
    buff_Pmax_C   = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(np.array([mcdc['global_tally_nuSigmaF']]), buff_nuSigmaF, MPI.SUM)
        if mcdc['technique']['IC_generator']:
            MPI.COMM_WORLD.Allreduce(np.array([mcdc['technique']['IC_tally_n']]), buff_IC_n, MPI.SUM)
            MPI.COMM_WORLD.Allreduce(np.array([mcdc['technique']['IC_tally_C']]), buff_IC_C, MPI.SUM)
            MPI.COMM_WORLD.Allreduce(np.array([mcdc['technique']['IC_Pmax_n']]), buff_Pmax_n, MPI.MAX)
            MPI.COMM_WORLD.Allreduce(np.array([mcdc['technique']['IC_Pmax_C']]), buff_Pmax_C, MPI.MAX)
    
    # IC generator: Increase number of active cycles?
    if mcdc['technique']['IC_generator']:
        Pmax_n     = buff_Pmax_n[0]
        Pmax_C     = buff_Pmax_C[0]
        Pmax       = max(Pmax_n, Pmax_C)
        N_inactive = mcdc['setting']['N_inactive']
        N_active   = mcdc['setting']['N_active']

        N_active_new = math.ceil(Pmax*N_active)
        if N_active_new > N_active:
            mcdc['technique']['IC_resample'] = True
            mcdc['setting']['N_active']      = N_active_new
            mcdc['setting']['N_cycle']       = N_inactive + N_active_new
            # Now the Pmax hold 1/w_prime (or P/w) for resampling
            Nn = mcdc['technique']['IC_N_neutron']
            Np = mcdc['technique']['IC_N_precursor']
            n  = mcdc['technique']['IC_n_eff'] # NOT using the new value
            p  = mcdc['technique']['IC_C_eff']
            mcdc['technique']['IC_Pmax_n'] = Nn/N_active_new/n
            mcdc['technique']['IC_Pmax_C'] = Np/N_active_new/p
        else:
            mcdc['technique']['IC_resample'] = False
            mcdc['technique']['IC_Pmax_n']   = 0.0
            mcdc['technique']['IC_Pmax_C']   = 0.0

    # Update and store k_eff
    mcdc['k_eff'] = buff_nuSigmaF[0]/N_particle
    mcdc['k_cycle'][i_cycle] = mcdc['k_eff']
    mcdc['technique']['IC_n_eff'] = buff_IC_n[0]
    mcdc['technique']['IC_C_eff'] = buff_IC_C[0]
   
    # Accumulate running average
    if mcdc['cycle_active']:
        mcdc['k_avg'] += mcdc['k_eff']
        mcdc['k_sdv'] += mcdc['k_eff']*mcdc['k_eff']

        N = 1 + mcdc['i_cycle'] - mcdc['setting']['N_inactive']
        mcdc['k_avg_running'] = mcdc['k_avg']/N
        if N == 1:
            mcdc['k_sdv_running'] = 0.0
        else:
            mcdc['k_sdv_running'] = \
                    math.sqrt((mcdc['k_sdv']/N - mcdc['k_avg_running']**2)\
                    /(N-1))

    # Reset accumulators
    mcdc['global_tally_nuSigmaF'] = 0.0
    mcdc['technique']['IC_tally_n'] = 0.0
    mcdc['technique']['IC_tally_C'] = 0.0

    # =====================================================================
    # Gyration radius
    # =====================================================================
    
    if mcdc['setting']['gyration_radius']:
        # Center of mass
        N_local     = mcdc['bank_census']['size']
        total_local = np.zeros(4, np.float64) # [x,y,z,W]
        total       = np.zeros(4, np.float64)
        for i in range(N_local):
            P = mcdc['bank_census']['particles'][i]
            total_local[0] += P['x']*P['w']
            total_local[1] += P['y']*P['w']
            total_local[2] += P['z']*P['w']
            total_local[3] += P['w']
        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(total_local, total, MPI.SUM)
        # COM
        W     = total[3]
        com_x = total[0]/W
        com_y = total[1]/W
        com_z = total[2]/W
    
        # Distance RMS
        rms_local = np.zeros(1, np.float64)
        rms       = np.zeros(1, np.float64)
        gr_type = mcdc['setting']['gyration_radius_type']
        if gr_type == GR_ALL:
            for i in range(N_local):
                P = mcdc['bank_census']['particles'][i]
                rms_local[0] += ((P['x'] - com_x)**2 + (P['y'] - com_y)**2 +\
                                 (P['z'] - com_z)**2)*P['w']
        elif gr_type == GR_INFINITE_X:
            for i in range(N_local):
                P = mcdc['bank_census']['particles'][i]
                rms_local[0] += ((P['y'] - com_y)**2 + (P['z'] - com_z)**2)\
                                *P['w']
        elif gr_type == GR_INFINITE_Y:
            for i in range(N_local):
                P = mcdc['bank_census']['particles'][i]
                rms_local[0] += ((P['x'] - com_x)**2 + (P['z'] - com_z)**2)\
                                *P['w']
        elif gr_type == GR_INFINITE_Z:
            for i in range(N_local):
                P = mcdc['bank_census']['particles'][i]
                rms_local[0] += ((P['x'] - com_x)**2 + (P['y'] - com_y)**2)\
                                *P['w']
        elif gr_type == GR_ONLY_X:
            for i in range(N_local):
                P = mcdc['bank_census']['particles'][i]
                rms_local[0] += ((P['x'] - com_x)**2)*P['w']
        elif gr_type == GR_ONLY_Y:
            for i in range(N_local):
                P = mcdc['bank_census']['particles'][i]
                rms_local[0] += ((P['y'] - com_y)**2)*P['w']
        elif gr_type == GR_ONLY_Z:
            for i in range(N_local):
                P = mcdc['bank_census']['particles'][i]
                rms_local[0] += ((P['z'] - com_z)**2)*P['w']

        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(rms_local, rms, MPI.SUM)
        rms = math.sqrt(rms[0]/W)
        
        # Gyration radius
        mcdc['gyration_radius'][i_cycle] = rms
            
#==============================================================================
# Move to event
#==============================================================================

@njit
def move_to_event(P, mcdc):
    # =========================================================================
    # Get distances to events
    # =========================================================================

    # Distance to nearest geometry boundary (surface or lattice)
    # Also set particle material and speed
    d_boundary, event_boundary = distance_to_boundary(P, mcdc)

    # Distance to tally mesh
    d_mesh = INF
    if mcdc['cycle_active']:
        d_mesh = distance_to_mesh(P, mcdc['tally']['mesh'], mcdc)

    # Distance to time boundary
    speed = get_particle_speed(P, mcdc)
    d_time_boundary = speed*(mcdc['setting']['time_boundary'] - P['t'])

    # Distance to census time
    idx           = mcdc['technique']['census_idx']
    d_time_census = speed*(mcdc['technique']['census_time'][idx] - P['t'])

    # Distance to collision
    d_collision = distance_to_collision(P, mcdc)

    # =========================================================================
    # Determine event
    #   Priority (in case of coincident events):
    #     boundary > time_boundary > mesh > collision
    # =========================================================================

    # Find the minimum
    event    = event_boundary
    distance = d_boundary
    if d_time_boundary*PREC < distance:
        event    = EVENT_TIME_BOUNDARY
        distance = d_time_boundary
    if d_time_census*PREC < distance:
        event    = EVENT_CENSUS
        distance = d_time_census
    if d_mesh*PREC < distance:
        event    = EVENT_MESH
        distance = d_mesh
    if d_collision*PREC < distance:
        event    = EVENT_COLLISION
        distance = d_collision

    # Crossing both boundary and mesh
    if d_boundary == d_mesh:
        # Surface and mesh?
        if event == EVENT_SURFACE:
            surface = mcdc['surfaces'][P['surface_ID']]
            event = EVENT_SURFACE_N_MESH
            #if not surface['reflective']:
            #    event = EVENT_SURFACE_N_MESH
        elif event == EVENT_SURFACE_MOVE:
            event = EVENT_SURFACE_MOVE_N_MESH
        # Lattice and mesh?
        elif event == EVENT_LATTICE:
            event = EVENT_LATTICE_N_MESH

    # Crossing both time census and mesh
    if event == EVENT_CENSUS and d_time_census == d_mesh:
        event = EVENT_CENSUS_N_MESH

    # Assign event
    P['event'] = event

    # Return if particle is already beyond current census time
    #if event == EVENT_CENSUS and distance < 0.0:
    #    return

    # =========================================================================
    # Move particle
    # =========================================================================

    # Score tracklength tallies
    if mcdc['tally']['tracklength'] and mcdc['cycle_active']:
        score_tracklength(P, distance, mcdc)
    if mcdc['setting']['mode_eigenvalue']:
        global_tally(P, distance, mcdc)

    # Move particle
    move_particle(P, distance, mcdc)

@njit
def distance_to_collision(P, mcdc):
    # Get total cross-section
    material = mcdc['materials'][P['material_ID']]
    SigmaT   = material['total'][P['g']]

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi     = rng(mcdc)
    distance  = -math.log(xi)/SigmaT
    return distance

@njit
def distance_to_boundary(P, mcdc):
    '''
    Find the nearest geometry boundary, which could be lattice or surface, and
    return the event type (EVENT_SURFACE or EVENT_LATTICE) and the distance

    We recursively check from the top level cell. If surface and lattice are 
    coincident, EVENT_SURFACE is prioritized.
    '''

    distance = INF
    event    = -1

    # Translation accumulator
    trans = np.zeros(3)

    # Top level cell
    cell = mcdc['cells'][P['cell_ID']]

    # Recursively check if cell is a lattice cell, until material cell is found
    while True:
        # Distance to nearest surface
        d_surface, surface_ID, surface_move = \
                distance_to_nearest_surface(P, cell, trans, mcdc)

        # Check if smaller
        if d_surface*PREC < distance:
            distance            = d_surface
            event               = EVENT_SURFACE
            P['surface_ID']     = surface_ID
            P['translation'][:] = trans

            if surface_move:
                event = EVENT_SURFACE_MOVE

        # Lattice cell?
        if cell['lattice']:
            # Get lattice
            lattice = mcdc['lattices'][cell['lattice_ID']]

            # Get lattice center for translation)
            trans -= cell['lattice_center']
           
            # Distance to lattice
            d_lattice = distance_to_lattice(P, lattice, trans)
            
            # Check if smaller
            if d_lattice*PREC < distance:
                distance        = d_lattice
                event           = EVENT_LATTICE
                P['surface_ID'] = -1

            # Get universe
            mesh        = lattice['mesh']
            x, y, z     = mesh_uniform_get_index(P, mesh, trans)
            universe_ID = lattice['universe_IDs'][x,y,z]

            # Update translation
            trans[0] -= mesh['x0'] + (x+0.5)*mesh['dx']
            trans[1] -= mesh['y0'] + (y+0.5)*mesh['dy']
            trans[2] -= mesh['z0'] + (z+0.5)*mesh['dz']

            # Get inner cell
            cell_ID = get_particle_cell(P, universe_ID, trans, mcdc)
            cell    = mcdc['cells'][cell_ID]

        else:
            # Material cell found, set material_ID
            P['material_ID'] = cell['material_ID']
            break

    return distance, event

@njit
def distance_to_nearest_surface(P, cell, trans, mcdc):
    distance     = INF
    surface_ID   = -1
    surface_move = False

    for i in range(cell['N_surface']):
        surface = mcdc['surfaces'][cell['surface_IDs'][i]]
        d, sm = surface_distance(P, surface, trans, mcdc)
        if d < distance:
            distance     = d
            surface_ID   = surface['ID']
            surface_move = sm
    return distance, surface_ID, surface_move

@njit
def distance_to_lattice(P, lattice, trans):
    mesh = lattice['mesh']

    x  = P['x'] + trans[0]
    y  = P['y'] + trans[1]
    z  = P['z'] + trans[2]
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']

    d = INF
    d = min(d, mesh_uniform_distance_search(x, ux, mesh['x0'], mesh['dx']))
    d = min(d, mesh_uniform_distance_search(y, uy, mesh['y0'], mesh['dy']))
    d = min(d, mesh_uniform_distance_search(z, uz, mesh['z0'], mesh['dz']))
    return d

@njit
def distance_to_mesh(P, mesh, mcdc):
    x  = P['x']
    y  = P['y']
    z  = P['z']
    t  = P['t']
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']
    v  = get_particle_speed(P, mcdc)

    d = INF
    d = min(d, mesh_distance_search(x, ux, mesh['x']))
    d = min(d, mesh_distance_search(y, uy, mesh['y']))
    d = min(d, mesh_distance_search(z, uz, mesh['z']))
    d = min(d, mesh_distance_search(t, 1.0/v, mesh['t']))
    return d

#==============================================================================
# Surface crossing
#==============================================================================

@njit
def surface_crossing(P, mcdc):
    trans = P['translation']

    # Implement BC
    surface = mcdc['surfaces'][P['surface_ID']]
    surface_bc(P, surface, trans)

    # Trigger mesh crossing?
    if surface['reflective'] and P['event'] == EVENT_SURFACE_N_MESH:
        mesh_crossing(P, mcdc)

    # Small shift to ensure crossing
    surface_shift(P, surface, trans, mcdc)
 
    # Check new cell?
    if P['alive'] and not surface['reflective']:
        cell  = mcdc['cells'][P['cell_ID']]
        if not cell_check(P, cell, trans, mcdc):
            trans = np.zeros(3)
            P['cell_ID'] = get_particle_cell(P, 0, trans, mcdc)

#==============================================================================
# Mesh crossing
#==============================================================================

@njit
def mesh_crossing(P, mcdc):
    # Tally mesh crossing
    if mcdc['tally']['crossing'] and mcdc['cycle_active']:
        mesh = mcdc['tally']['mesh']

        # Determine which dimension is crossed
        x, y, z, t, flag = mesh_crossing_evaluate(P, mesh)

        # Score on tally
        if flag == MESH_X and mcdc['tally']['crossing_x']:
            score_crossing_x(P, t, x, y, z, mcdc)
        if flag == MESH_Y and mcdc['tally']['crossing_y']:
            score_crossing_y(P, t, x, y, z, mcdc)
        if flag == MESH_Z and mcdc['tally']['crossing_z']:
            score_crossing_z(P, t, x, y, z, mcdc)
        if flag == MESH_T and mcdc['tally']['crossing_t']:
            score_crossing_t(P, t, x, y, z, mcdc)
    
    # Shift particle if only mesh crossing occurs
    if P['event'] == EVENT_MESH:
        shift_particle(P, SHIFT)

#==============================================================================
# Collision
#==============================================================================

@njit
def collision(P, mcdc):
    # Get the reaction cross-sections
    material = mcdc['materials'][P['material_ID']]
    g        = P['g']
    SigmaT   = material['total'][g]
    SigmaC   = material['capture'][g]
    SigmaS   = material['scatter'][g]
    SigmaF   = material['fission'][g]

    if mcdc['technique']['implicit_capture']:
        P['w'] *= (SigmaT-SigmaC)/SigmaT
        SigmaT      -= SigmaC

    # Sample collision type
    xi = rng(mcdc)*SigmaT
    tot = SigmaS
    if tot > xi:
        event = EVENT_SCATTERING
    else:
        tot += SigmaF
        if tot > xi:
            event = EVENT_FISSION
        else:
            event = EVENT_CAPTURE
    P['event'] = event

#==============================================================================
# Capture
#==============================================================================

@njit
def capture(P, mcdc):
    # Kill the current particle
    P['alive'] = False

    pass

#==============================================================================
# Scattering
#==============================================================================

@njit
def scattering(P, mcdc):
    # Get outgoing spectrum
    material = mcdc['materials'][P['material_ID']]
    g        = P['g']
    chi_s    = material['chi_s'][g]
    nu_s     = material['nu_s'][g]
    G        = material['G']

    # Get effective and new weight
    if mcdc['technique']['weighted_emission']:
        weight_eff = P['w']
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P['w']

    # Kill the current particle
    P['alive'] = False

    # Get number of secondaries
    N = int(math.floor(weight_eff*nu_s + rng(mcdc)))

    for n in range(N):
        # Create new particle
        P_new = np.zeros(1, dtype=type_.particle_record)[0]

        # Copy relevant attributes
        P_new['x'] = P['x']         
        P_new['y'] = P['y']         
        P_new['z'] = P['z']         
        P_new['t'] = P['t']      

        # Set weight
        P_new['w'] = weight_new

        # Sample outgoing energy
        xi  = rng(mcdc)
        tot = 0.0
        for g_out in range(G):
            tot += chi_s[g_out]
            if tot > xi:
                break
        P_new['g'] = g_out
        
        # Sample scattering angle
        mu = 2.0*rng(mcdc) - 1.0;
        
        # Sample azimuthal direction
        azi     = 2.0*PI*rng(mcdc)
        cos_azi = math.cos(azi)
        sin_azi = math.sin(azi)
        Ac      = (1.0 - mu**2)**0.5

        ux = P['ux']
        uy = P['uy']
        uz = P['uz']
        
        if uz != 1.0:
            B = (1.0 - P['uz']**2)**0.5
            C = Ac/B
            
            P_new['ux'] = ux*mu + (ux*uz*cos_azi - uy*sin_azi)*C
            P_new['uy'] = uy*mu + (uy*uz*cos_azi + ux*sin_azi)*C
            P_new['uz'] = uz*mu - cos_azi*Ac*B
        
        # If dir = 0i + 0j + k, interchange z and y in the scattering formula
        else:
            B = (1.0 - uy**2)**0.5
            C = Ac/B
            
            P_new['ux'] = ux*mu + (ux*uy*cos_azi - uz*sin_azi)*C
            P_new['uz'] = uz*mu + (uz*uy*cos_azi + ux*sin_azi)*C
            P_new['uy'] = uy*mu - cos_azi*Ac*B

        # Bank
        add_particle(P_new, mcdc['bank_active'])
        
#==============================================================================
# Fission
#==============================================================================

@njit
def fission(P, mcdc):
    # Get constants
    material = mcdc['materials'][P['material_ID']]
    G        = material['G']
    J        = material['J']
    g        = P['g']
    nu_p     = material['nu_p'][g]
    nu       = nu_p
    if J>0: 
        nu_d  = material['nu_d'][g]
        nu   += sum(nu_d)

    # Get effective and new weight
    if mcdc['technique']['weighted_emission']:
        weight_eff = P['w']
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P['w']

    # Kill the current particle
    P['alive'] = False

    # Sample prompt and delayed neutrons
    for jj in range(J+1):
        prompt = jj == 0
        j      = jj - 1

        # Get data (average emission number, spectrum, decay rate)
        if prompt:
            nu       = nu_p
            spectrum = material['chi_p'][g]
        else:
            nu       = nu_d[j]
            spectrum = material['chi_d'][j]
            decay    = material['decay'][j]

        # Sample number of fission neutrons
        N = int(math.floor(weight_eff*nu/mcdc['k_eff'] + rng(mcdc)))

        # Push fission neutrons to bank
        for n in range(N):
            # Create new particle
            P_new = np.zeros(1, dtype=type_.particle_record)[0]

            # Copy relevant attributes
            P_new['x'] = P['x']         
            P_new['y'] = P['y']         
            P_new['z'] = P['z']         
            P_new['t'] = P['t']      

            # Set weight
            P_new['w'] = weight_new

            # Sample emission time
            if not prompt:
                xi = rng(mcdc)
                P_new['t'] -= math.log(xi)/decay

                # Skip if it's beyond time boundary
                if P_new['t'] > mcdc['setting']['time_boundary']:
                    continue

            # Sample outgoing energy
            xi  = rng(mcdc)
            tot = 0.0
            for g_out in range(G):
                tot += spectrum[g_out]
                if tot > xi:
                    break
            P_new['g'] = g_out

            # Sample isotropic direction
            P_new['ux'], P_new['uy'], P_new['uz'] = \
                    sample_isotropic_direction(mcdc)

            # Bank
            if mcdc['setting']['mode_eigenvalue']:
                add_particle(P_new, mcdc['bank_census'])
            else:
                add_particle(P_new, mcdc['bank_active'])

#==============================================================================
# Branchless collision
#==============================================================================

@njit
def branchless_collision(P, mcdc):
    # Data
    material = mcdc['materials'][P['material_ID']]
    w        = P['w']
    g        = P['g']
    SigmaT   = material['total'][g]
    SigmaF   = material['fission'][g]
    SigmaS   = material['scatter'][g]
    nu_s     = material['nu_s'][g]
    nu_p     = material['nu_p'][g]/mcdc['k_eff']
    nu_d     = material['nu_d'][g]/mcdc['k_eff']
    J        = material['J']
    G        = material['G']

    # Total nu fission
    nu = nu_p
    for j in range(J):
        nu += nu_d[j]

    # Set weight
    n_scatter    = nu_s*SigmaS
    n_fission    = nu*SigmaF
    n_total      = n_fission + n_scatter
    P['w'] *= n_total/SigmaT

    # Set spectrum and decay rate
    fission = True
    prompt  = True
    if rng(mcdc) < n_scatter/n_total:
        fission  = False
        spectrum = material['chi_s'][g]
    else:
        xi  = rng(mcdc)*nu
        tot = nu_p
        if xi < tot:
            spectrum = material['chi_p'][g]
        else:
            prompt = False
            for j in range(J):
                tot += nu_d[j]
                if xi < tot:
                    spectrum = material['chi_d'][j]
                    decay    = material['decay'][j]
                    break

    # Set time
    if not prompt:
        xi = rng(mcdc)
        P['t'] -= math.log(xi)/decay

        # Kill if it's beyond time boundary
        if P['t'] > mcdc['setting']['time_boundary']:
            P['alive'] = False
            return
    
    # Set energy
    xi  = rng(mcdc)
    tot = 0.0
    for g_out in range(G):
        tot += spectrum[g_out]
        if tot > xi:
            P['g'] = g_out
            break

    # Set direction (TODO: anisotropic scattering)
    P['ux'], P['uy'], P['uz'] = sample_isotropic_direction(mcdc)

#==============================================================================
# Time boundary
#==============================================================================

@njit
def time_boundary(P, mcdc):
    P['alive'] = False

#==============================================================================
# Weight widow
#==============================================================================
    
@njit
def weight_window(P, mcdc):
    # Get indices
    t, x, y, z, outside = mesh_get_index(P, mcdc['technique']['ww_mesh'])

    # Target weight
    w_target = mcdc['technique']['ww'][t,x,y,z]
   
    # Surviving probability
    p = P['w']/w_target

    # Set target weight
    P['w'] = w_target

    # If above target
    if p > 1.0:
        # Splitting (keep the original particle)
        n_split = math.floor(p)
        for i in range(n_split-1):
            add_particle(copy_particle(P), mcdc['bank_active'])

        # Russian roulette
        p -= n_split
        xi = rng(mcdc)
        if xi <= p:
            add_particle(copy_particle(P), mcdc['bank_active'])

    # Below target
    else:
        # Russian roulette
        xi = rng(mcdc)
        if xi > p:
            P['alive'] = False

#==============================================================================
# Miscellany
#==============================================================================

@njit
def binary_search(val, grid):
    """
    Binary search that returns the bin index of a value val given grid grid
    
    Some special cases:
        val < min(grid)  --> -1
        val > max(grid)  --> size of bins
        val = a grid point --> bin location whose upper bound is val
                                 (-1 if val = min(grid)
    """
    
    left  = 0
    right = len(grid) - 1
    mid   = -1
    while left <= right:
        mid = (int((left + right)/2))
        if grid[mid] < val: left = mid + 1
        else:            right = mid - 1
    return int(right)
