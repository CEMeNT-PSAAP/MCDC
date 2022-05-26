import math
import numba as nb

import mcdc.type_ as type_

from mcdc.class_.point import Point
from mcdc.constant     import *
from mcdc.print_       import print_error

#==============================================================================
# Random sampling
#==============================================================================

@nb.njit
def sample_isotropic_direction(rng):
    # Sample polar cosine and azimuthal angle uniformly
    mu  = 2.0*rng.random() - 1.0
    azi = 2.0*PI*rng.random()

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2)**0.5
    y = math.cos(azi)*c
    z = math.sin(azi)*c
    x = mu
    return x, y, z

@nb.njit
def sample_uniform(a, b, rng):
    return a + rng.random() * (b - a)

@nb.njit
def sample_discrete(p, rng):
    tot = 0.0
    xi  = rng.random()
    for i in range(p.shape[0]):
        tot += p[i]
        if tot > xi:
            return i

#==============================================================================
# Particle bank operations
#==============================================================================

@nb.njit
def add_particle(P, bank):
    if bank['size'] == bank['max_size']:
        with nb.objmode():
            print_error('Particle bank "'+bank['tag']+
                        '" exceeds its maximum size.')
    bank['particles'][bank['size']] = P
    bank['size'] += 1

@nb.njit
def pop_particle(bank):
    if bank['size'] == 0:
        with nb.objmode():
            print_error('Particle bank "'+bank['tag']+'" is empty.')
    bank['size'] -= 1
    P = bank['particles'][bank['size']]
    return P.copy()

#==============================================================================
# Set cell
#==============================================================================

@nb.njit
def set_cell(P, mcdc):
    found = False
    for C in mcdc.cells:
        if C.test_point(P):
            P['cell_ID'] = C.ID
            found = True

            # Set particle speed
            P['speed'] = C.material.speed[P['group']]

            break
    if not found:
        print("A particle is lost at (",P['position']['x'],P['position']['y'],P['position']['z'],")")
        P['alive'] = False

#==============================================================================
# Move to event
#==============================================================================

@nb.njit
def move_to_event(P, mcdc):
    # Get distances to events
    d_collision        = collision_distance(P, mcdc)
    surface, d_surface = surface_distance(P, mcdc)
    d_mesh             = mcdc.tally.mesh.distance(P, mcdc)
    d_time_boundary    = P['speed']*(mcdc.setting.time_boundary - P['time'])

    # Determine event
    event, distance = determine_event(d_collision, d_surface, d_time_boundary,
                                      d_mesh)

    # Score tracklength tallies
    if mcdc.tally.tracklength:
        score_tracklength(P, distance, mcdc)

    # Move particle
    move_particle(P, distance)

    # Record surface if crossed
    if event == EVENT_SURFACE:
        P['surface_ID'] = surface.ID
        # Also mesh crossing?
        if d_surface == d_mesh and not surface.reflective:
            event = EVENT_SURFACE_N_MESH
    else:
        P['surface_ID'] = -1
    
    return event

@nb.njit
def collision_distance(P, mcdc):
    # Get total cross-section
    SigmaT = mcdc.cells[P['cell_ID']].material.total[P['group']]

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Time absorption?
    if mcdc.setting.mode_alpha:
        SigmaT += abs(mcdc.tally_global.alpha_eff)/P['speed']

    # Sample collision distance
    xi     = mcdc.rng.random()
    distance  = -math.log(xi)/SigmaT
    return distance

@nb.njit
def surface_distance(P, mcdc):
    surface  = None
    distance = INF

    cell = mcdc.cells[P['cell_ID']]
    for S in cell.surfaces:
        d = S.distance(P)
        if d < distance:
            surface  = S
            distance = d
    return surface, distance

@nb.njit
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

@nb.njit
def score_tracklength(P, distance, mcdc):
    mcdc.tally.score_tracklength(P, distance)

    # Score eigenvalue tallies
    if mcdc.setting.mode_eigenvalue:
        material = mcdc.cells[P['cell_ID']].material
        g        = P['group']
        weight   = P['weight']
        nu       = material.nu_p[g]\
                   + sum(material.nu_d[g])
        SigmaF   = material.fission[g]
        nuSigmaF = nu*SigmaF
        mcdc.tally_global.nuSigmaF += weight*distance*nuSigmaF

        if mcdc.setting.mode_alpha:
            mcdc.tally_global.inverse_speed += weight*distance/P['speed']

@nb.njit
def move_particle(P, distance):
    P['x']    += P['ux']*distance
    P['y']    += P['uy']*distance
    P['z']    += P['uz']*distance
    P['time'] += distance/P['speed']


#==============================================================================
# Surface crossing
#==============================================================================

@nb.njit
def surface_crossing(P, mcdc):
    # Implement BC
    surface = mcdc.surfaces[P['surface_ID']]
    surface.apply_bc(P)

    # Small kick to make sure crossing
    move_particle(P, PRECISION)
 
    # Set new cell
    if P['alive'] and not surface.reflective:
        set_cell(P, mcdc)

#==============================================================================
# Collision
#==============================================================================

@nb.njit
def collision(P, mcdc):
    # Kill the current particle
    P['alive'] = False

    # Get the reaction cross-sections
    material = mcdc.cells[P['cell_ID']].material
    g        = P['group']
    SigmaT   = material.total[g]
    SigmaC   = material.capture[g]
    SigmaS   = material.scatter[g]
    SigmaF   = material.fission[g]

    if mcdc.setting.mode_alpha:
        Sigma_alpha = abs(mcdc.tally_global.alpha_eff)/P['speed']
        SigmaT += Sigma_alpha

    if mcdc.setting.implicit_capture:
        if mcdc.setting.mode_alpha:
            P['weight'] *= (SigmaT-SigmaC-Sigma_alpha)/SigmaT
            SigmaT      -= (SigmaC + Sigma_alpha)
        else:
            P['weight'] *= (SigmaT-SigmaC)/SigmaT
            SigmaT      -= SigmaC

    # Sample collision type
    xi = mcdc.rng.random()*SigmaT
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

@nb.njit
def capture(P, mcdc):
    pass

#==============================================================================
# Scattering
#==============================================================================

@nb.njit
def scattering(P, mcdc):
    # Get outgoing spectrum
    material = mcdc.cells[P['cell_ID']].material
    g        = P['group']
    chi_s    = material.chi_s[g]
    nu_s     = material.nu_s[g]
    G        = material.G

    # Get effective and new weight
    if mcdc.setting.implicit_capture:
        weight_eff = 1.0
        weight_new = P['weight']
    else:
        weight_eff = P['weight']
        weight_new = 1.0

    N = int(math.floor(weight_eff*nu_s + mcdc.rng.random()))

    for n in range(N):
        # Copy particle (need to revive)
        P_new = P.copy()
        P_new['alive'] = True

        # Set weight
        P_new['weight'] = weight_new

        # Sample outgoing energy
        xi  = mcdc.rng.random()
        tot = 0.0
        for g_out in range(G):
            tot += chi_s[g_out]
            if tot > xi:
                break
        P_new['group'] = g_out
        P_new['speed'] = material.speed[g_out]
        
        # Sample scattering angle
        # TODO: anisotropic scattering
        mu = 2.0*mcdc.rng.random() - 1.0;
        
        # Sample azimuthal direction
        azi     = 2.0*PI*mcdc.rng.random()
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
        add_particle(P_new, mcdc.bank_history)
        
#==============================================================================
# Fission
#==============================================================================

@nb.njit
def fission(P, mcdc):
    # Get constants
    material = mcdc.cells[P['cell_ID']].material
    G        = material.G
    J        = material.J
    g        = P['group']
    weight   = P['weight']
    nu_p     = material.nu_p[g]
    nu       = nu_p
    if J>0: 
        nu_d  = material.nu_d[g]
        nu   += sum(nu_d)

    # Get effective and new weight
    if mcdc.setting.implicit_capture:
        weight_eff = 1.0
        weight_new = weight
    else:
        weight_eff = weight
        weight_new = 1.0

    # Sample number of fission neutrons
    #   in fixed-source, k_eff = 1.0
    N = int(math.floor(weight_eff*nu/mcdc.tally_global.k_eff 
                          + mcdc.rng.random()))

    # Push fission neutrons to bank
    for n in range(N):
        # Copy particle (need to revive)
        P_new = P.copy()
        P_new['alive'] = True

        # Set weight
        P_new['weight'] = weight_new

        # Determine if it's prompt or delayed neutrons, 
        # then get the energy spectrum and decay constant
        xi  = mcdc.rng.random()*nu
        tot = nu_p
        # Prompt?
        if tot > xi:
            spectrum = material.chi_p[g]
        else:
            # Which delayed group?
            for j in range(J):
                tot += nu_d[j]
                if tot > xi:
                    spectrum = material.chi_d[j]
                    decay    = material.decay[j]
                    break

            # Sample emission time
            xi = mcdc.rng.random()
            P_new['time'] = P['time'] - math.log(xi)/decay

            # Skip if it's beyond time boundary
            if P_new['time'] > mcdc.setting.time_boundary:
                continue

        # Sample outgoing energy
        xi  = mcdc.rng.random()
        tot = 0.0
        for g_out in range(G):
            tot += spectrum[g_out]
            if tot > xi:
                break
        P_new['group'] = g_out
        P_new['speed'] = material.speed[g_out]

        # Sample isotropic direction
        P_new['ux'], P_new['uy'], P_new['uz'] = \
                sample_isotropic_direction(mcdc.rng)

        # Bank
        add_particle(P_new, mcdc.bank_fission)

#==============================================================================
# Time reaction
#==============================================================================

@nb.njit
def time_reaction(P, mcdc):
    if mcdc.tally_global.alpha_eff > 0:
        pass # Already killed
    else:
        P_new = P.copy()
        mcdc.bank.history.append(P_new)

#==============================================================================
# Time boundary
#==============================================================================

@nb.njit
def time_boundary(P, mcdc):
    P['alive'] = False

    # Check if mesh crossing occured
    mesh_crossing(P, mcdc)

#==============================================================================
# Mesh crossing
#==============================================================================

@nb.njit
def mesh_crossing(P, mcdc):
    if not mcdc.tally.crossing:
        # Small-kick to ensure crossing
        move_particle(P, PRECISION)
    else:
        # Use small-kick back and forth to determine which mesh is crossed
        # First, backward small-kick
        move_particle(P, -PRECISION)
        t1, x1, y1, z1 = mcdc.tally.mesh.get_index(P)
        # Then, double forward small-kick
        move_particle(P, 2*PRECISION)
        t2, x2, y2, z2 = mcdc.tally.mesh.get_index(P)

        # Determine which mesh is crossed
        crossing_x = False
        crossing_t = False
        if x1 != x2:
            crossing_x = True
        if t1 != t2:
            crossing_t = True

        # Score on tally
        if crossing_x and mcdc.tally.crossing_x:
            mcdc.tally.score_crossing_x(P, t1, x1, y1, z1)
        if crossing_t and mcdc.tally.crossing_t:
            mcdc.tally.score_crossing_t(P, t1, x1, y1, z1)

#==============================================================================
# Miscellany
#==============================================================================

'''
@nb.njit
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
'''
