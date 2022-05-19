import math

from numba import njit

from mcdc.class_.point import Point
from mcdc.constant     import *

#==============================================================================
# Random sampling
#==============================================================================

@njit
def sample_isotropic_direction(rng):
    # Sample polar cosine and azimuthal angle uniformly
    mu  = 2.0*rng.random() - 1.0
    azi = 2.0*PI*rng.random()

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2)**0.5
    y = math.cos(azi)*c
    z = math.sin(azi)*c
    x = mu
    return Point(x, y, z)

@njit
def sample_uniform(a, b, rng):
    return a + rng.random() * (b - a)

@njit
def sample_discrete(p, rng):
    tot = 0.0
    xi  = rng.random()
    for i in range(p.shape[0]):
        tot += p[i]
        if tot > xi:
            return i

#==============================================================================
# Set cell
#==============================================================================

@njit
def set_cell(P, mcdc):
    found = False
    for C in mcdc.cells:
        if C.test_point(P):
            P.cell = C
            found = True
            break
    if not found:
        print("A particle is lost at (",P.position.x,P.position.y,P.position.z,")")
        P.alive = False

#==============================================================================
# Move to event
#==============================================================================

@njit
def move_to_event(P, mcdc):
    # Get speed
    P.speed = P.cell.material.speed[P.group]

    # Get distances to events
    d_collision        = collision_distance(P, mcdc)
    surface, d_surface = surface_distance(P)
    d_time_boundary    = P.speed*(mcdc.setting.time_boundary - P.time)
    d_mesh             = mcdc.tally.mesh.distance(P)

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
        P.surface = surface
        # Also mesh crossing?
        if d_surface == d_mesh and not surface.reflective:
            event = EVENT_SURFACE_N_MESH
    
    return event

@njit
def collision_distance(P, mcdc):
    # Get total XS
    SigmaT = P.cell.material.total[P.group]

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Time absorption?
    if mcdc.setting.mode_alpha:
        SigmaT += abs(mcdc.tally_global.alpha_eff)/P.speed

    # Sample collision distance
    xi     = mcdc.rng.random()
    distance  = -math.log(xi)/SigmaT
    return distance

@njit
def surface_distance(P):
    surface  = None
    distance = INF
    for S in P.cell.surfaces:
        d = S.distance(P)
        if d < distance:
            surface  = S
            distance = d
    return surface, distance

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

@njit
def score_tracklength(P, distance, mcdc):
    mcdc.tally.score_tracklength(P, distance)

    # Score eigenvalue tallies
    if mcdc.setting.mode_eigenvalue:
        nu       = P.cell.material.nu_p[P.group]\
                   + sum(P.cell.material.nu_d[P.group])
        SigmaF   = P.cell.material.fission[P.group]
        nuSigmaF = nu*SigmaF
        mcdc.tally_global.nuSigmaF += P.weight*distance*nuSigmaF

        if mcdc.setting.mode_alpha:
            mcdc.tally_global.inverse_speed += P.weight*distance\
                                       /P.cell.material.speed[P.group]

@njit
def move_particle(P, distance):
    P.position.x += P.direction.x*distance
    P.position.y += P.direction.y*distance
    P.position.z += P.direction.z*distance
    P.time       += distance/P.speed


#==============================================================================
# Surface crossing
#==============================================================================

@njit
def surface_crossing(P, mcdc):
    # Implement BC
    P.surface.apply_bc(P)

    # Small kick to make sure crossing
    move_particle(P, PRECISION)
 
    # Set new cell
    if P.alive and not P.surface.reflective:
        set_cell(P, mcdc)

#==============================================================================
# Collision
#==============================================================================

@njit
def collision(P, mcdc):
    # Kill the current particle
    P.alive = False

    SigmaT = P.cell.material.total[P.group]
    SigmaC = P.cell.material.capture[P.group]
    SigmaS = P.cell.material.scatter[P.group]
    SigmaF = P.cell.material.fission[P.group]

    if mcdc.setting.mode_alpha:
        Sigma_alpha = abs(mcdc.tally_global.alpha_eff)/P.speed
        SigmaT += Sigma_alpha

    if mcdc.setting.implicit_capture:
        if mcdc.setting.mode_alpha:
            P.weight *= (SigmaT-SigmaC-Sigma_alpha)/SigmaT
            SigmaT   -= (SigmaC + Sigma_alpha)
        else:
            P.weight *= (SigmaT-SigmaC)/SigmaT
            SigmaT   -= SigmaC

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

@njit
def capture(P, mcdc):
    pass

#==============================================================================
# Scattering
#==============================================================================

@njit
def scattering(P, mcdc):
    # Ger outgoing spectrum
    chi_s = P.cell.material.chi_s[P.group]
    nu_s  = P.cell.material.nu_s[P.group]
    G     = P.cell.material.G

    # Get effective and new weight
    if mcdc.setting.implicit_capture:
        weight_eff = 1.0
        weight_new = P.weight
    else:
        weight_eff = P.weight
        weight_new = 1.0

    N = int(math.floor(weight_eff*nu_s + mcdc.rng.random()))

    for n in range(N):
        # Create copy
        P_new = P.copy()
        P_new.weight = weight_new

        # Sample outgoing energy
        xi  = mcdc.rng.random()
        tot = 0.0
        for g_out in range(G):
            tot += chi_s[g_out]
            if tot > xi:
                break
        P_new.group = g_out
        
        # Sample scattering angle
        # TODO: anisotropic scattering
        mu = 2.0*mcdc.rng.random() - 1.0;
        
        # Sample azimuthal direction
        azi     = 2.0*PI*mcdc.rng.random()
        cos_azi = math.cos(azi)
        sin_azi = math.sin(azi)
        Ac      = (1.0 - mu**2)**0.5

        ux = P_new.direction.x
        uy = P_new.direction.y
        uz = P_new.direction.z
        
        if uz != 1.0:
            B = (1.0 - P.direction.z**2)**0.5
            C = Ac/B
            
            P_new.direction.x = ux*mu + (ux*uz*cos_azi - uy*sin_azi)*C
            P_new.direction.y = uy*mu + (uy*uz*cos_azi + ux*sin_azi)*C
            P_new.direction.z = uz*mu - cos_azi*Ac*B
        
        # If dir = 0i + 0j + k, interchange z and y in the scattering formula
        else:
            B = (1.0 - uy**2)**0.5
            C = Ac/B
            
            P_new.direction.x = ux*mu + (ux*uy*cos_azi - uz*sin_azi)*C
            P_new.direction.z = uz*mu + (uz*uy*cos_azi + ux*sin_azi)*C
            P_new.direction.y = uy*mu - cos_azi*Ac*B
        
        # Bank
        mcdc.bank.history.append(P_new)
        
#==============================================================================
# Fission
#==============================================================================

@njit
def fission(P, mcdc):
    # Get group numbers
    G = P.cell.material.G
    J = P.cell.material.J
    
    # Total nu
    nu_p = P.cell.material.nu_p[P.group]
    nu = nu_p
    if J>0: 
        nu_d = P.cell.material.nu_d[P.group]
        nu += sum(nu_d)

    # Get effective and new weight
    if mcdc.setting.implicit_capture:
        weight_eff = 1.0
        weight_new = P.weight
    else:
        weight_eff = P.weight
        weight_new = 1.0

    # Sample number of fission neutrons
    #   in fixed-source, k_eff = 1.0
    N = int(math.floor(weight_eff*nu/mcdc.tally_global.k_eff 
                          + mcdc.rng.random()))

    # Push fission neutrons to bank
    for n in range(N):
        # Create copy
        P_new = P.copy()
        P_new.weight = weight_new

        # Determine if it's prompt or delayed neutrons, 
        # then get the energy spectrum and decay constant
        xi  = mcdc.rng.random()*nu
        tot = nu_p
        # Prompt?
        if tot > xi:
            spectrum = P.cell.material.chi_p[P.group]
        else:
            # Which delayed group?
            for j in range(J):
                tot += nu_d[j]
                if tot > xi:
                    spectrum = P.cell.material.chi_d[j]
                    decay    = P.cell.material.decay[j]
                    break

            # Sample emission time
            xi = mcdc.rng.random()
            P_new.time = P.time - math.log(xi)/decay

            # Skip if it's beyond time boundary
            if P_new.time > mcdc.setting.time_boundary:
                continue

        # Sample outgoing energy
        xi  = mcdc.rng.random()
        tot = 0.0
        for g_out in range(G):
            tot += spectrum[g_out]
            if tot > xi:
                break
        P_new.group = g_out

        # Sample isotropic direction
        P_new.direction = sample_isotropic_direction(mcdc.rng)

        # Bank
        mcdc.bank.fission.append(P_new)

#==============================================================================
# Time reaction
#==============================================================================

@njit
def time_reaction(P, mcdc):
    if mcdc.tally_global.alpha_eff > 0:
        pass # Already killed
    else:
        P_new = P.copy()
        mcdc.bank.history.append(P_new)

#==============================================================================
# Time boundary
#==============================================================================

@njit
def time_boundary(P, mcdc):
    P.alive = False

    # Check if mesh crossing occured
    mesh_crossing(P, mcdc)

#==============================================================================
# Mesh crossing
#==============================================================================

@njit
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
'''
