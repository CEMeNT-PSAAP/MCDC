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
    else:
        ux = source['ux']
        uy = source['uy']
        uz = source['uz']

    # Energy and time
    group = sample_discrete(source['group'], rng)
    time  = sample_uniform(source['time'][0], source['time'][1], rng)

    P          = make_particle()
    P['x']     = x
    P['y']     = y
    P['z']     = z
    P['ux']    = ux
    P['uy']    = uy
    P['uz']    = uz
    P['group'] = group
    P['time']  = time

    return P

#==============================================================================
# Particle bank operations
#==============================================================================

@njit
def add_particle(P, bank):
    if bank['size'] == bank['particles'].shape[0]:
        with objmode():
            print_error('Particle bank is full.')
    bank['particles'][bank['size']] = P
    bank['size'] += 1

@njit
def pop_particle(bank):
    if bank['size'] == 0:
        with objmode():
            print_error('Particle bank is empty.')
    bank['size'] -= 1
    P = bank['particles'][bank['size']]
    return copy_particle(P)

@njit
def manage_particle_banks(mcdc):
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
        rng_rebase(mcdc)
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

@njit
def population_control(mcdc):
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
        P['weight'] *= td
        add_particle(P, bank_source)

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
def normalize_weight(bank, norm):
    # Get total weight
    W = total_weight(bank)

    # Normalize weight
    for P in bank['particles']:
        P['weight'] *= norm/W

@njit
def total_weight(bank):
    # Local total weight
    W_local = np.zeros(1)
    for i in range(bank['size']):
        W_local[0] += bank['particles'][i]['weight']
    
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
                    dtype=type_.particle)

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

# TODO
'''
@njit
def bank_IC(P, mcdc):
    material = get_material(P, mcdc)
    g        = P['group']
    J        = material['J']
    G        = material['G']
    nu_d     = material['nu_d'][g]/mcdc['k_eff']
    SigmaT   = material['total'][g]
    SigmaF   = material['fission'][g]
    decay    = material['decay']
    v        = P['speed']
    weight   = P['weight']

    flux = weight/SigmaT
    
    # Sample prompt neutron
    prob = flux/v/mcdc['IC_n_total']
    if rng(mcdc) < prob:
        P_new           = copy_particle(P)
        P_new['time']   = 0.0
        P_new['weight'] = 1.0
        add_particle(P_new, mcdc['bank_IC'])
        mcdc['IC_counter_p'] += 1
        mcdc['IC_fission']   += SigmaF*v

    # Sample delayed neutrons
    tmax = mcdc['setting']['IC_tmax']
    for j in range(J):
        prob = flux*nu_d[j]*SigmaF/decay[j]/mcdc['IC_n_total']\
               *(1.0 - math.exp(-decay[j]*tmax))
        if rng(mcdc) < prob:
            P_new           = copy_particle(P)
            P_new['weight'] = 1.0

            # Rejection-sample emission time
            while True:
                xi   = rng(mcdc)
                time = -math.log(xi)/decay[j]
                # Accept if it's inside maximum time
                if time < tmax:
                    break
            P_new['time'] = time

            # Sample energy
            spectrum = material['chi_d'][j]
            xi       = rng(mcdc)
            tot      = 0.0
            for g_out in range(G):
                tot += spectrum[g_out]
                if tot > xi:
                    break
            P_new['group'] = g_out
            P_new['speed'] = material['speed'][g_out]

            # Sample isotropic direction
            P_new['ux'], P_new['uy'], P_new['uz'] = \
                    sample_isotropic_direction(mcdc)
            
            add_particle(P_new, mcdc['bank_IC'])
            mcdc['IC_counter_d'][j] += 1
'''

#==============================================================================
# Particle operations
#==============================================================================

@njit
def make_particle():
    P = np.zeros(1, dtype=type_.particle)[0]
    P['x']           = 0.0
    P['y']           = 0.0
    P['z']           = 0.0
    P['ux']          = 0.0
    P['uy']          = 0.0
    P['uz']          = 0.0
    P['group']       = 0
    P['time']        = 0.0
    P['weight']      = 1.0
    P['alive']       = True
    P['speed']       = 1.0
    P['cell_ID']     = -1
    P['surface_ID']  = -1
    P['universe_ID'] = -1
    P['event']       = -1
    P['shift_x']     = 0.0
    P['shift_y']     = 0.0
    P['shift_z']     = 0.0
    P['shift_t']     = 0.0
    return P

@njit
def copy_particle(P):
    P_new = np.zeros(1, dtype=type_.particle)[0]
    P_new['x']           = P['x']         
    P_new['y']           = P['y']         
    P_new['z']           = P['z']         
    P_new['ux']          = P['ux']        
    P_new['uy']          = P['uy']        
    P_new['uz']          = P['uz']        
    P_new['group']       = P['group']     
    P_new['time']        = P['time']      
    P_new['weight']      = P['weight']    
    P_new['alive']       = P['alive']     
    P_new['speed']       = P['speed']     
    P_new['cell_ID']     = P['cell_ID']   
    P_new['surface_ID']  = P['surface_ID']
    P_new['universe_ID'] = P['universe_ID']
    P_new['event']       = P['event']
    P_new['shift_x']     = P['shift_x']
    P_new['shift_y']     = P['shift_y']
    P_new['shift_z']     = P['shift_z']
    P_new['shift_t']     = P['shift_t']
    return P_new

@njit
def get_universe(P, mcdc):
    return mcdc['universes'][P['universe_ID']]

@njit
def get_cell(P, mcdc):
    return mcdc['cells'][P['cell_ID']]

@njit
def get_material(P, mcdc):
    cell = get_cell(P, mcdc)
    return mcdc['materials'][cell['material_ID']]

@njit
def move_particle(P, distance):
    P['x']    += P['ux']*distance
    P['y']    += P['uy']*distance
    P['z']    += P['uz']*distance
    P['time'] += distance/P['speed']

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
    P['time'] += shift

#==============================================================================
# Universe operations
#==============================================================================

@njit
def set_universe(P, mcdc):
    # Get lattice and mesh
    lattice = mcdc['lattice']
    mesh    = lattice['mesh']

    # Get mesh index
    t, x, y, z = mesh_get_index(P, mesh)

    # Set universe
    P['universe_ID'] = lattice['universe_IDs'][t,x,y,z]

    # Set particle shift
    P['shift_x'] = -0.5*(mesh['x'][x] + mesh['x'][x+1])
    P['shift_y'] = -0.5*(mesh['y'][y] + mesh['y'][y+1])
    P['shift_z'] = -0.5*(mesh['z'][z] + mesh['z'][z+1])
    P['shift_t'] = -0.5*(mesh['t'][t] + mesh['t'][t+1])

    # Set cell
    set_cell(P, mcdc)

#==============================================================================
# Cell operations
#==============================================================================

@njit
def set_cell(P, mcdc):
    universe = get_universe(P, mcdc)
    for cell_ID in universe['cell_IDs']:
        cell = mcdc['cells'][cell_ID]
        if cell_check(P, cell, mcdc):
            # Set cell ID
            P['cell_ID'] = cell['ID']
            
            # Set particle speed
            material   = mcdc['materials'][cell['material_ID']]
            P['speed'] = material['speed'][P['group']]
            
            return
    print("A particle is lost at (",P['x'],P['y'],P['z'],")")
    P['alive'] = False

@njit
def cell_check(P, cell, mcdc):
    for i in range(cell['N_surface']):
        surface = mcdc['surfaces'][cell['surface_IDs'][i]]
        result  = surface_evaluate(P, surface)
        if cell['positive_flags'][i]:
            if result < 0.0: return False
        else:
            if result > 0.0: return False
    return True

#==============================================================================
# Surface operations
#==============================================================================
"""
Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
"""

@njit
def surface_evaluate(P, surface):
    x = P['x'] + P['shift_x']
    y = P['y'] + P['shift_y']
    z = P['z'] + P['shift_z']
    
    G = surface['G']
    H = surface['H']
    I = surface['I']
    J = surface['J']

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
def surface_bc(P, surface):
    if surface['vacuum']:
        P['alive'] = False
    elif surface['reflective']:
        surface_reflect(P, surface)

@njit
def surface_reflect(P, surface):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']
    nx, ny, nz  = surface_normal(P, surface)
    c  = 2.0*(nx*ux + ny*uy + nz*uz) # 2.0*surface_normal_component(...)

    P['ux'] = ux - c*nx
    P['uy'] = uy - c*ny
    P['uz'] = uz - c*nz

@njit
def surface_shift(P, surface):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']

    # Get surface normal
    nx, ny, nz = surface_normal(P, surface)

    # The shift
    shift_x = nx*SHIFT
    shift_y = ny*SHIFT
    shift_z = nz*SHIFT

    # Get dot product to determine shift sign
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
def surface_normal(P, surface):
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
    x = P['x'] + P['shift_x']
    y = P['y'] + P['shift_y']
    z = P['z'] + P['shift_z']
    
    dx = 2*A*x + D*y + E*z + G
    dy = 2*B*y + D*x + F*z + H
    dz = 2*C*z + E*x + F*y + I
    
    norm = (dx**2 + dy**2 + dz**2)**0.5
    return dx/norm, dy/norm, dz/norm
    
@njit
def surface_normal_component(P, surface):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']
    nx, ny, nz  = surface_normal(P, surface)
    return nx*ux + ny*uy + nz*uz

@njit
def surface_distance(P, surface):
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']

    G  = surface['G']
    H  = surface['H']
    I  = surface['I']

    if surface['linear']:
        distance = -surface_evaluate(P, surface)/(G*ux + H*uy + I*uz)
        # Moving away from the surface
        if distance < 0.0: return INF
        else:              return distance
        
    x  = P['x'] + P['shift_x']
    y  = P['y'] + P['shift_y']
    z  = P['z'] + P['shift_z']

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
    c = surface_evaluate(P, surface)
    
    determinant = b*b - 4.0*a*c
    
    # Roots are complex  : no intersection
    # Roots are identical: tangent
    # ==> return huge number
    if determinant <= 0.0:
        return INF
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
        return min(root_1, root_2)

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
    dist = (grid[idx] - value)/direction
    return dist

@njit
def mesh_get_index(P, mesh):
    t = binary_search(P['time'], mesh['t'])
    x = binary_search(P['x'],    mesh['x'])
    y = binary_search(P['y'],    mesh['y'])
    z = binary_search(P['z'],    mesh['z'])
    return t, x, y, z

@njit
def mesh_crossing_evaluate(P, mesh):
    # Shift backward
    shift_particle(P, -SHIFT)
    t1, x1, y1, z1 = mesh_get_index(P, mesh)
   
    # Double shift forward
    shift_particle(P, 2*SHIFT)
    t2, x2, y2, z2 = mesh_get_index(P, mesh)

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

@njit 
def mesh_crossing_shift(P, flag):
    if flag == MESH_X:
        if P['ux'] > 0.0:
            P['x'] += SHIFT
        else:
            P['x'] -= SHIFT
    elif flag == MESH_Y:
        if P['uy'] > 0.0:
            P['y'] += SHIFT
        else:
            P['y'] -= SHIFT
    elif flag == MESH_Z:
        if P['uz'] > 0.0:
            P['z'] += SHIFT
        else:
            P['z'] -= SHIFT
    elif flag == MESH_T:
        P['time'] += SHIFT

#==============================================================================
# Tally operations
#==============================================================================

@njit
def score_tracklength(P, distance, mcdc):
    tally    = mcdc['tally']
    material = get_material(P, mcdc)

    # Get indices
    g = P['group']
    t, x, y, z = mesh_get_index(P, tally['mesh'])

    # Score
    flux = distance*P['weight']
    if tally['flux']:
        score_flux(g, t, x, y, z, flux, tally['score']['flux'])
    if tally['fission']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['fission'])
    if tally['density']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['density'])
    if tally['current']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current'])
    if tally['eddington']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington'])

@njit
def score_crossing_x(P, t, x, y, z, mcdc):
    tally    = mcdc['tally']
    material = get_material(P, mcdc)

    # Get indices
    g = P['group']
    if P['ux'] > 0.0:
        x += 1

    # Score
    flux = P['weight']/abs(P['ux'])
    if tally['flux_x']:
        score_flux(g, t, x, y, z, flux, tally['score']['flux_x'])
    if tally['fission_x']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['fission_x'])
    if tally['density_x']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['density_x'])
    if tally['current_x']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_x'])
    if tally['eddington_x']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_x'])

@njit
def score_crossing_y(P, t, x, y, z, mcdc):
    tally    = mcdc['tally']
    material = get_material(P, mcdc)

    # Get indices
    g = P['group']
    if P['uy'] > 0.0:
        y += 1

    # Score
    flux = P['weight']/abs(P['ux'])
    if tally['flux_y']:
        score_flux(g, t, x, y, z, flux, tally['score']['flux_y'])
    if tally['fission_y']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['fission_y'])
    if tally['density_y']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['density_y'])
    if tally['current_y']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_y'])
    if tally['eddington_y']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_y'])

@njit
def score_crossing_z(P, t, x, y, z, mcdc):
    tally    = mcdc['tally']
    material = get_material(P, mcdc)

    # Get indices
    g = P['group']
    if P['uz'] > 0.0:
        z += 1

    # Score
    flux = P['weight']/abs(P['ux'])
    if tally['flux_z']:
        score_flux(g, t, x, y, z, flux, tally['score']['flux_z'])
    if tally['fission_z']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['fission_z'])
    if tally['density_z']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['density_z'])
    if tally['current_z']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_z'])
    if tally['eddington_z']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_z'])

@njit
def score_crossing_t(P, t, x, y, z, mcdc):
    tally    = mcdc['tally']
    material = get_material(P, mcdc)
    
    # Get indices
    g  = P['group']
    t += 1

    # Score
    flux = P['weight']*P['speed']
    if tally['flux_t']:
        score_flux(g, t, x, y, z, flux, tally['score']['flux_t'])
    if tally['fission_t']:
        flux *= material['fission'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['fission_t'])
    if tally['density_t']:
        flux /= material['speed'][g]
        score_flux(g, t, x, y, z, flux, tally['score']['density_t'])
    if tally['current_t']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_t'])
    if tally['eddington_t']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_t'])

@njit
def score_flux(g, t, x, y, z, flux, score):
    score['bin'][g, t, x, y, z] += flux

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
    material = get_material(P, mcdc)

    # Parameters
    flux     = distance*P['weight']
    g        = P['group']
    nu       = material['nu_p'][g] + sum(material['nu_d'][g])
    SigmaF   = material['fission'][g]
    nuSigmaF = nu*SigmaF

    mcdc['fission_production'] += flux*nuSigmaF

    # TODO
    '''
    if mcdc['setting']['generate_IC'] and mcdc['2nd_last_iteration']:
        J     = material['J']
        nu_d  = material['nu_d'][g]/mcdc['k_eff']
        decay = material['decay']

        # Prompt
        IC_n_total = 1.0/P['speed']
        # Delayed
        tmax = mcdc['setting']['IC_tmax']
        for j in range(J):
            IC_n_total += nu_d[j]*SigmaF/decay[j]\
                          *(1.0 - math.exp(-decay[j]*tmax))
        # Finalize
        IC_n_total *= flux
        mcdc['IC_n_total'] += IC_n_total
    '''

@njit
def global_tally_closeout_history(mcdc):
    N_particle = mcdc['setting']['N_particle']

    if mcdc['setting']['mode_eigenvalue']:
        i_cycle = mcdc['i_cycle']

        # MPI Allreduce
        buff1 = np.zeros(1, np.float64)
        buff3 = np.zeros(1, np.float64)
        with objmode():
            MPI.COMM_WORLD.Allreduce(np.array([mcdc['fission_production']]), buff1, MPI.SUM)
            # TODO
            '''
            if mcdc['setting']['generate_IC'] and mcdc['2nd_last_iteration']:
                MPI.COMM_WORLD.Allreduce(np.array([mcdc['IC_n_total']]), buff2, MPI.SUM)
            '''
        mcdc['fission_production'] = buff1[0]
        #mcdc['IC_n_total']    = buff2[0]/N_particle
        
        # Update and store k_eff
        mcdc['k_eff'] = mcdc['fission_production']/N_particle
        mcdc['k_cycle'][i_cycle] = mcdc['k_eff']
       
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
        mcdc['fission_production'] = 0.0

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
                total_local[0] += P['x']*P['weight']
                total_local[1] += P['y']*P['weight']
                total_local[2] += P['z']*P['weight']
                total_local[3] += P['weight']
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
                                     (P['z'] - com_z)**2)*P['weight']
            elif gr_type == GR_INFINITE_X:
                for i in range(N_local):
                    P = mcdc['bank_census']['particles'][i]
                    rms_local[0] += ((P['y'] - com_y)**2 + (P['z'] - com_z)**2)\
                                    *P['weight']
            elif gr_type == GR_INFINITE_Y:
                for i in range(N_local):
                    P = mcdc['bank_census']['particles'][i]
                    rms_local[0] += ((P['x'] - com_x)**2 + (P['z'] - com_z)**2)\
                                    *P['weight']
            elif gr_type == GR_INFINITE_Z:
                for i in range(N_local):
                    P = mcdc['bank_census']['particles'][i]
                    rms_local[0] += ((P['x'] - com_x)**2 + (P['y'] - com_y)**2)\
                                    *P['weight']
            elif gr_type == GR_ONLY_X:
                for i in range(N_local):
                    P = mcdc['bank_census']['particles'][i]
                    rms_local[0] += ((P['x'] - com_x)**2)*P['weight']
            elif gr_type == GR_ONLY_Y:
                for i in range(N_local):
                    P = mcdc['bank_census']['particles'][i]
                    rms_local[0] += ((P['y'] - com_y)**2)*P['weight']
            elif gr_type == GR_ONLY_Z:
                for i in range(N_local):
                    P = mcdc['bank_census']['particles'][i]
                    rms_local[0] += ((P['z'] - com_z)**2)*P['weight']

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
    # Get distances to events
    d_collision           = distance_to_collision(P, mcdc)
    d_surface, surface_ID = distance_to_nearest_surface(P, mcdc)
    d_mesh                = distance_to_mesh(P, mcdc['tally']['mesh'])
    d_lattice             = distance_to_mesh(P, mcdc['lattice']['mesh'])
    d_time_boundary       = P['speed']*(mcdc['setting']['time_boundary'] - P['time'])

    # Determine event
    event    = EVENT_COLLISION
    distance = d_collision
    if distance > d_time_boundary:
        event = EVENT_TIME_BOUNDARY
        distance = d_time_boundary
    if distance > d_surface:
        event  = EVENT_SURFACE
        distance = d_surface
    if distance > d_lattice:
        event  = EVENT_LATTICE
        distance = d_lattice
    if distance > d_mesh:
        event  = EVENT_MESH
        distance = d_mesh

    # Score tracklength tallies
    if mcdc['tally']['tracklength'] and mcdc['cycle_active']:
        score_tracklength(P, distance, mcdc)
    if mcdc['setting']['mode_eigenvalue']:
        global_tally(P, distance, mcdc)

    # Move particle
    move_particle(P, distance)

    # Record surface if crossed
    if event == EVENT_SURFACE:
        P['surface_ID'] = surface_ID
        # Also mesh crossing?
        surface = mcdc['surfaces'][P['surface_ID']]
        if d_surface == d_mesh and not surface['reflective']:
            event = EVENT_SURFACE_N_MESH
    else:
        P['surface_ID'] = -1

    # Lattice and mesh?
    if event == EVENT_LATTICE and d_lattice == d_mesh:
        event = EVENT_LATTICE_N_MESH
    
    # Assign event
    P['event'] = event

@njit
def distance_to_collision(P, mcdc):
    # Get total cross-section
    material = get_material(P, mcdc)
    SigmaT   = material['total'][P['group']]

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi     = rng(mcdc)
    distance  = -math.log(xi)/SigmaT
    return distance

@njit
def distance_to_nearest_surface(P, mcdc):
    surface_ID = -1
    distance   = INF

    cell = get_cell(P, mcdc)
    for i in range(cell['N_surface']):
        surface = mcdc['surfaces'][cell['surface_IDs'][i]]
        d = surface_distance(P, surface)
        if d < distance:
            surface_ID = surface['ID']
            distance   = d
    return distance, surface_ID

@njit
def distance_to_mesh(P, mesh):
    x  = P['x']
    y  = P['y']
    z  = P['z']
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']
    t  = P['time']
    v  = P['speed']

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
    # Implement BC
    surface = mcdc['surfaces'][P['surface_ID']]
    surface_bc(P, surface)

    # Small shift to ensure crossing
    surface_shift(P, surface)
 
    # Set new cell
    if P['alive'] and not surface['reflective']:
        set_cell(P, mcdc)

#==============================================================================
# Mesh crossing
#==============================================================================

@njit
def mesh_crossing(P, mcdc):
    mesh = mcdc['tally']['mesh']

    # Determine which dimension is crossed
    x, y, z, t, flag = mesh_crossing_evaluate(P, mesh)

    # Shift particle if surface and lattice are not crossed as well
    if P['event'] == EVENT_MESH:
        mesh_crossing_shift(P, flag)

    # Tally mesh crossing
    if mcdc['tally']['crossing'] and mcdc['cycle_active']:
        # Score on tally
        if flag == MESH_X and mcdc['tally']['crossing_x']:
            score_crossing_x(P, t, x, y, z, mcdc)
        if flag == MESH_Y and mcdc['tally']['crossing_y']:
            score_crossing_y(P, t, x, y, z, mcdc)
        if flag == MESH_Z and mcdc['tally']['crossing_z']:
            score_crossing_z(P, t, x, y, z, mcdc)
        if flag == MESH_T and mcdc['tally']['crossing_t']:
            score_crossing_t(P, t, x, y, z, mcdc)

#==============================================================================
# Lattice crossing
#==============================================================================

@njit 
def lattice_crossing(P, mcdc):
    lattice = mcdc['lattice']
    mesh    = lattice['mesh']

    # Determine which dimension is crossed
    x, y, z, t, flag = mesh_crossing_evaluate(P, mesh)

    # Apply BC if crossed
    reflected = False
    if flag == MESH_X:
        if x == 0 and P['ux'] < 0.0:
            if lattice['reflective_x-']:
                reflected = True
                P['ux'] *= -1
            else:
                P['alive'] = False
        elif x == len(mesh['x'])-2 and P['ux'] > 0.0:
            if lattice['reflective_x+']:
                reflected = True
                P['ux'] *= -1
            else:
                P['alive'] = False
    elif flag == MESH_Y:
        if y == 0 and P['uy'] < 0.0:
            if lattice['reflective_y-']:
                reflected = True
                P['uy'] *= -1
            else:
                P['alive'] = False
        elif y == len(mesh['y'])-2 and P['uy'] > 0.0:
            if lattice['reflective_y+']:
                reflected = True
                P['uy'] *= -1
            else:
                P['alive'] = False
    elif flag == MESH_Z:
        if z == 0 and P['uz'] < 0.0:
            if lattice['reflective_z-']:
                reflected = True
                P['uz'] *= -1
            else:
                P['alive'] = False
        elif z == len(mesh['z'])-2 and P['uz'] > 0.0:
            if lattice['reflective_z+']:
                reflected = True
                P['uz'] *= -1
            else:
                P['alive'] = False

    # Shift particle
    mesh_crossing_shift(P, flag)

    # Set new universe
    if P['alive'] and not reflected:
        set_universe(P, mcdc)

#==============================================================================
# Collision
#==============================================================================

@njit
def collision(P, mcdc):
    # Kill the current particle
    P['alive'] = False

    # Get the reaction cross-sections
    material = get_material(P, mcdc)
    g        = P['group']
    SigmaT   = material['total'][g]
    SigmaC   = material['capture'][g]
    SigmaS   = material['scatter'][g]
    SigmaF   = material['fission'][g]

    if mcdc['technique']['implicit_capture']:
        P['weight'] *= (SigmaT-SigmaC)/SigmaT
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
    return event

#==============================================================================
# Capture
#==============================================================================

@njit
def capture(P, mcdc):
    pass

#==============================================================================
# Scattering
#==============================================================================

@njit
def scattering(P, mcdc):
    # Get outgoing spectrum
    material = get_material(P, mcdc)
    g        = P['group']
    chi_s    = material['chi_s'][g]
    nu_s     = material['nu_s'][g]
    G        = material['G']

    # Get effective and new weight
    weight = P['weight']
    if mcdc['technique']['weighted_emission']:
        weight_eff = weight
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = weight

    N = int(math.floor(weight_eff*nu_s + rng(mcdc)))

    for n in range(N):
        # Copy particle (need to revive)
        P_new = copy_particle(P)
        P_new['alive'] = True

        # Set weight
        P_new['weight'] = weight_new

        # Sample outgoing energy
        xi  = rng(mcdc)
        tot = 0.0
        for g_out in range(G):
            tot += chi_s[g_out]
            if tot > xi:
                break
        P_new['group'] = g_out
        P_new['speed'] = material['speed'][g_out]
        
        # Sample scattering angle
        mu = 2.0*rng(mcdc) - 1.0;
        
        # Sample azimuthal direction
        azi     = 2.0*PI*rng(mcdc)
        cos_azi = math.cos(azi)
        sin_azi = math.sin(azi)
        Ac      = (1.0 - mu**2)**0.5

        ux = P_new['ux']
        uy = P_new['uy']
        uz = P_new['uz']
        
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
    material = get_material(P, mcdc)
    G        = material['G']
    J        = material['J']
    g        = P['group']
    weight   = P['weight']
    nu_p     = material['nu_p'][g]
    nu       = nu_p
    if J>0: 
        nu_d  = material['nu_d'][g]
        nu   += sum(nu_d)

    # Get effective and new weight
    if mcdc['technique']['weighted_emission']:
        weight_eff = weight
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = weight

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
            # Copy particle (need to revive)
            P_new = copy_particle(P)
            P_new['alive'] = True

            # Set weight
            P_new['weight'] = weight_new

            # Sample emission time
            if not prompt:
                xi = rng(mcdc)
                P_new['time'] -= math.log(xi)/decay

                # Skip if it's beyond time boundary
                if P_new['time'] > mcdc['setting']['time_boundary']:
                    continue

            # Sample outgoing energy
            xi  = rng(mcdc)
            tot = 0.0
            for g_out in range(G):
                tot += spectrum[g_out]
                if tot > xi:
                    break
            P_new['group'] = g_out
            P_new['speed'] = material['speed'][g_out]

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
    material = get_material(P, mcdc)
    w        = P['weight']
    g        = P['group']
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
    P['weight'] *= n_total/SigmaT

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
        P['time'] -= math.log(xi)/decay

        # Kill if it's beyond time boundary
        if P['time'] > mcdc['setting']['time_boundary']:
            P['alive'] = False
            return
    
    # Set energy
    xi  = rng(mcdc)
    tot = 0.0
    for g_out in range(G):
        tot += spectrum[g_out]
        if tot > xi:
            P['group'] = g_out
            P['speed'] = material['speed'][g_out]
            break

    # Set direction (TODO: anisotropic scattering)
    P['ux'], P['uy'], P['uz'] = sample_isotropic_direction(mcdc)

#==============================================================================
# Time boundary
#==============================================================================

@njit
def time_boundary(P, mcdc):
    P['alive'] = False

    # Check if mesh crossing occured
    mesh_crossing(P, mcdc)

#==============================================================================
# Move to event
#==============================================================================
    
@njit
def weight_window(P, mcdc):
    # Get indices
    t, x, y, z = mesh_get_index(P, mcdc['technique']['ww_mesh'])

    # Target weight
    w_target = mcdc['technique']['ww'][t,x,y,z]
   
    # Surviving probability
    p = P.weight/w_target

    # Set target weight
    P.weight = w_target

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
            P.alive = False

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
