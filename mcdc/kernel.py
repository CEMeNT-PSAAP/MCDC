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

@njit
# TODO: use cummulative density function and binary search
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
    g         = int(mcdc['rng_g'])
    c         = int(mcdc['rng_c'])
    g_new     = 1
    c_new     = 0
    mod       = int(mcdc['rng_mod'])
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
    g        = int(mcdc['rng_g'])
    c        = int(mcdc['rng_c'])
    mod      = int(mcdc['rng_mod'])
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
    bank['particles'][bank['size']] = P
    bank['size'] += 1

@njit
def pop_particle(bank):
    if bank['size'] == 0:
        with objmode():
            print_error('Particle bank "'+bank['tag']+'" is empty.')
    bank['size'] -= 1
    P = bank['particles'][bank['size']]
    return copy_particle(P)

@njit
def manage_particle_banks(mcdc):
    if mcdc['setting']['mode_eigenvalue']:
        # Normalize weight
        normalize_weight(mcdc['bank_census'], mcdc['setting']['N_hist'])

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
    M           = mcdc['setting']['N_hist']
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

#==============================================================================
# Particle operations
#==============================================================================

@njit
def make_particle():
    P = np.zeros(1, dtype=type_.particle)[0]
    P['x']          = 0.0
    P['y']          = 0.0
    P['z']          = 0.0
    P['ux']         = 0.0
    P['uy']         = 0.0
    P['uz']         = 0.0
    P['group']      = 0
    P['time']       = 0.0
    P['weight']     = 1.0
    P['alive']      = True
    P['speed']      = 1.0
    P['cell_ID']    = -1
    P['surface_ID'] = -1
    return P

@njit
def copy_particle(P):
    P_new = np.zeros(1, dtype=type_.particle)[0]
    P_new['x']          = P['x']         
    P_new['y']          = P['y']         
    P_new['z']          = P['z']         
    P_new['ux']         = P['ux']        
    P_new['uy']         = P['uy']        
    P_new['uz']         = P['uz']        
    P_new['group']      = P['group']     
    P_new['time']       = P['time']      
    P_new['weight']     = P['weight']    
    P_new['alive']      = P['alive']     
    P_new['speed']      = P['speed']     
    P_new['cell_ID']    = P['cell_ID']   
    P_new['surface_ID'] = P['surface_ID']
    return P_new

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

#==============================================================================
# Cell operations
#==============================================================================

@njit
def set_cell(P, mcdc):
    for cell in mcdc['cells']:
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
    for i in range(cell['N_surfaces']):
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
    x = P['x']
    y = P['y']
    z = P['z']
    
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
    x = P['x']
    y = P['y']
    z = P['z']
    
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
        
    x  = P['x']
    y  = P['y']
    z  = P['z']

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
        sqrt  = np.sqrt(determinant)
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

#==============================================================================
# Tally operations
#==============================================================================

@njit
def score_tracklength(P, distance, mcdc):
    tally = mcdc['tally']

    # Get indices
    g = P['group']
    t, x, y, z = mesh_get_index(P, tally['mesh'])

    # Score
    flux = distance*P['weight']
    if tally['flux']:
        score_flux(g, t, x, y, z, flux, tally['score']['flux'])
    if tally['current']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current'])
    if tally['eddington']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington'])

    # Score eigenvalue tallies
    if mcdc['setting']['mode_eigenvalue']:
        material = get_material(P, mcdc)
        g        = P['group']
        weight   = P['weight']
        nu       = material['nu_p'][g]\
                   + sum(material['nu_d'][g])
        SigmaF   = material['fission'][g]
        nuSigmaF = nu*SigmaF
        mcdc['nuSigmaF'] += weight*distance*nuSigmaF

        if mcdc['setting']['mode_alpha']:
            mcdc['inverse_speed'] += weight*distance/P['speed']

@njit
def score_crossing_x(P, t, x, y, z, mcdc):
    tally = mcdc['tally']

    # Get indices
    g = P['group']
    if P['ux'] > 0.0:
        x += 1

    # Score
    flux = P['weight']/abs(P['ux'])
    if tally['flux_x']:
        score_flux(g, t, x, y, z, flux, tally['score']['flux_x'])
    if tally['current_x']:
        score_current(g, t, x, y, z, flux, P, tally['score']['current_x'])
    if tally['eddington_x']:
        score_eddington(g, t, x, y, z, flux, P, tally['score']['eddington_x'])

@njit
def score_crossing_t(P, t, x, y, z, mcdc):
    tally = mcdc['tally']
    
    # Get indices
    g  = P['group']
    t += 1

    # Score
    flux = P['weight']*P['speed']
    if tally['flux_t']:
        score_flux(g, t, x, y, z, flux, tally['score']['flux_t'])
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
def tally_closeout_history(mcdc):
    tally = mcdc['tally']

    for name in literal_unroll(score_list):
        if tally[name]:
            score_closeout_history(tally['score'][name])

@njit
def score_closeout_history(score):
    # Accumulate sums of history
    score['sum'][:]    += score['bin']
    score['sum_sq'][:] += np.square(score['bin'])

    # Reset bin
    score['bin'].fill(0.0)

@njit
def tally_closeout(mcdc):
    tally = mcdc['tally']

    for name in literal_unroll(score_list):
        if tally[name]:
            score_closeout(tally['score'][name], mcdc['setting']['N_hist'], mcdc['i_iter'], mcdc)

    # Global tally
    N_hist = mcdc['setting']['N_hist']
    i_iter = mcdc['i_iter']

    if mcdc['setting']['mode_eigenvalue']:
        # MPI Allreduce
        buff1 = np.zeros(1, np.float64)
        buff2 = np.zeros(1, np.float64)
        with objmode():
            MPI.COMM_WORLD.Allreduce(np.array([mcdc['nuSigmaF']]), buff1, MPI.SUM)
            if mcdc['setting']['mode_alpha']:
                MPI.COMM_WORLD.Allreduce(np.array([mcdc['inverse_speed']]), buff2, MPI.SUM)
        mcdc['nuSigmaF'] = buff1[0]
        mcdc['inverse_speed'] = buff2[0]
        
        # Update and store k_eff
        mcdc['k_eff'] = mcdc['nuSigmaF']/N_hist
        mcdc['k_iterate'][i_iter] = mcdc['k_eff']
        
        # Update and store alpha_eff
        if mcdc['setting']['mode_alpha']:
            k_eff         = mcdc['k_eff']
            inverse_speed = mcdc['inverse_speed']/N_hist

            mcdc['alpha_eff'] += (k_eff - 1.0)/inverse_speed
            mcdc['alpha_iterate'][i_iter] = mcdc['alpha_eff']
                    
        # Reset accumulators
        mcdc['nuSigmaF'] = 0.0
        if mcdc['setting']['mode_alpha']:
            mcdc['inverse_speed'] = 0.0        

@njit
def score_closeout(score, N_hist, i_iter, mcdc):
    # MPI Reduce
    buff    = np.zeros_like(score['sum'])
    buff_sq = np.zeros_like(score['sum_sq'])
    with objmode():
        MPI.COMM_WORLD.Reduce(np.array(score['sum']), buff, MPI.SUM, 0)
        MPI.COMM_WORLD.Reduce(np.array(score['sum_sq']), buff_sq, MPI.SUM, 0)
    score['sum'][:]    = buff
    score['sum_sq'][:] = buff_sq
    
    # Store results
    score['mean'][i_iter,:] = score['sum']/N_hist
    score['sdev'][i_iter,:] = np.sqrt((score['sum_sq']/N_hist 
                                - np.square(score['mean'][i_iter]))\
                               /(N_hist-1))
    
    # Reset history sums
    score['sum'].fill(0.0)
    score['sum_sq'].fill(0.0)

#==============================================================================
# Move to event
#==============================================================================

@njit
def move_to_event(P, mcdc):
    # Get distances to events
    d_collision           = distance_to_collision(P, mcdc)
    d_surface, surface_ID = distance_to_nearest_surface(P, mcdc)
    d_mesh                = distance_to_mesh(P, mcdc)
    d_time_boundary       = P['speed']*(mcdc['setting']['time_boundary'] - P['time'])

    # Determine event
    event, distance = determine_event(d_collision, d_surface, d_time_boundary,
                                      d_mesh)

    # Score tracklength tallies
    if mcdc['tally']['tracklength']:
        score_tracklength(P, distance, mcdc)

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
    
    return event

@njit
def distance_to_collision(P, mcdc):
    # Get total cross-section
    material = get_material(P, mcdc)
    SigmaT   = material['total'][P['group']]

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Time absorption?
    if mcdc['setting']['mode_alpha']:
        SigmaT += abs(mcdc['alpha_eff'])/P['speed']

    # Sample collision distance
    xi     = rng(mcdc)
    distance  = -math.log(xi)/SigmaT
    return distance

@njit
def distance_to_nearest_surface(P, mcdc):
    surface_ID = -1
    distance   = INF

    cell = get_cell(P, mcdc)
    for i in range(cell['N_surfaces']):
        surface = mcdc['surfaces'][cell['surface_IDs'][i]]
        d = surface_distance(P, surface)
        if d < distance:
            surface_ID = surface['ID']
            distance   = d
    return distance, surface_ID

@njit
def distance_to_mesh(P, mcdc):
    x  = P['x']
    y  = P['y']
    z  = P['z']
    ux = P['ux']
    uy = P['uy']
    uz = P['uz']
    t  = P['time']
    v  = P['speed']

    mesh = mcdc['tally']['mesh']

    d = INF
    d = min(d, mesh_distance_search(x, ux, mesh['x']))
    d = min(d, mesh_distance_search(y, uy, mesh['y']))
    d = min(d, mesh_distance_search(z, uz, mesh['z']))
    d = min(d, mesh_distance_search(t, 1.0/v, mesh['t']))
    return d

@njit
def determine_event(d_collision, d_surface, d_time_boundary, d_mesh):
    event  = EVENT_COLLISION
    distance = d_collision
    if distance > d_time_boundary:
        event = EVENT_TIME_BOUNDARY
        distance = d_time_boundary
    if distance > d_surface:
        event  = EVENT_SURFACE
        distance = d_surface
    if distance > d_mesh:
        event  = EVENT_MESH
        distance = d_mesh
    return event, distance

#==============================================================================
# Surface crossing
#==============================================================================

@njit
def surface_crossing(P, mcdc):
    # Implement BC
    surface = mcdc['surfaces'][P['surface_ID']]
    surface_bc(P, surface)

    # Small kick to make sure crossing
    move_particle(P, PRECISION)
 
    # Set new cell
    if P['alive'] and not surface['reflective']:
        set_cell(P, mcdc)

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

    if mcdc['setting']['mode_alpha']:
        Sigma_alpha = abs(mcdc['alpha_eff'])/P['speed']
        SigmaT += Sigma_alpha

    if mcdc['technique']['implicit_capture']:
        if mcdc['setting']['mode_alpha']:
            P['weight'] *= (SigmaT-SigmaC-Sigma_alpha)/SigmaT
            SigmaT      -= (SigmaC + Sigma_alpha)
        else:
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
            tot += SigmaC
            if tot > xi:
                event = EVENT_CAPTURE
            else:
                event = EVENT_TIME_REACTION
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
        add_particle(P_new, mcdc['bank_history'])
        
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

    # Sample number of fission neutrons
    #   in fixed-source, k_eff = 1.0
    N = int(math.floor(weight_eff*nu/mcdc['k_eff'] + rng(mcdc)))

    # Push fission neutrons to bank
    for n in range(N):
        # Copy particle (need to revive)
        P_new = copy_particle(P)
        P_new['alive'] = True

        # Set weight
        P_new['weight'] = weight_new

        # Determine if it's prompt or delayed neutrons, 
        # then get the energy spectrum and decay constant
        xi  = rng(mcdc)*nu
        tot = nu_p
        # Prompt?
        if tot > xi:
            spectrum = material['chi_p'][g]
        else:
            # Which delayed group?
            for j in range(J):
                tot += nu_d[j]
                if tot > xi:
                    spectrum = material['chi_d'][j]
                    decay    = material['decay'][j]
                    break

            # Sample emission time
            xi = rng(mcdc)
            P_new['time'] = P['time'] - math.log(xi)/decay

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
            add_particle(P_new, mcdc['bank_history'])

#==============================================================================
# Time reaction
#==============================================================================

@njit
def time_reaction(P, mcdc):
    if mcdc['alpha_eff'] > 0:
        pass # Already killed
    else:
        add_particle(copy_particle(P), mcdc['bank_history'])

#==============================================================================
# Time boundary
#==============================================================================

@njit
def time_boundary(P, mcdc):
    P['alive'] = False

    # Check if mesh crossing occured
    mesh_crossing(P, mcdc)

#==============================================================================
# Mesh crossing
#==============================================================================

@njit
def mesh_crossing(P, mcdc):
    if not mcdc['tally']['crossing']:
        # Small-kick to ensure crossing
        move_particle(P, PRECISION)
    else:
        # Use small-kick back and forth to determine which mesh is crossed
        # First, backward small-kick
        move_particle(P, -PRECISION)
        t1, x1, y1, z1 = mesh_get_index(P, mcdc['tally']['mesh'])
        # Then, double forward small-kick
        move_particle(P, 2*PRECISION)
        t2, x2, y2, z2 = mesh_get_index(P, mcdc['tally']['mesh'])

        # Determine which mesh is crossed
        crossing_x = False
        crossing_t = False
        if x1 != x2:
            crossing_x = True
        if t1 != t2:
            crossing_t = True

        # Score on tally
        if crossing_x and mcdc['tally']['crossing_x']:
            score_crossing_x(P, t1, x1, y1, z1, mcdc)
        if crossing_t and mcdc['tally']['crossing_t']:
            score_crossing_t(P, t1, x1, y1, z1, mcdc)

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
            add_particle(copy_particle(P), mcdc['bank_history'])

        # Russian roulette
        p -= n_split
        xi = rng(mcdc)
        if xi <= p:
            add_particle(copy_particle(P), mcdc['bank_history'])

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
