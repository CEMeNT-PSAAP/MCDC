from   math  import floor
import numpy as     np

from mcdc.class_.particle import *
from mcdc.constant        import *

# Get mcdc global variables/objects
import mcdc.global_ as mcdc

#==============================================================================
# Isotropic direction
#==============================================================================
    
def isotropic_direction():
    # Sample polar cosine and azimuthal angle uniformly
    mu  = 2.0*mcdc.rng() - 1.0
    azi = 2.0*PI*mcdc.rng()

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2)**0.5
    y = np.cos(azi)*c
    z = np.sin(azi)*c
    x = mu
    return Point(x, y, z)

#==============================================================================
# Get source
#==============================================================================

def get_source():
    # Sample source
    xi = mcdc.rng()
    tot = 0.0
    source = None
    for s in mcdc.sources:
        tot += s.prob
        if xi < tot:
            source = s
            break
    P = source.get_particle()
    set_cell(P)
    set_census_time_idx(P)
    return P

#==============================================================================
# Set cell
#==============================================================================

def set_cell(P):
    cell     = None
    position = P.position
    time     = P.time
    for C in mcdc.cells:
        if C.test_point(position,time):
            cell = C
            break
    if cell == None:
        print_error("A particle is lost at "+str(position))
        sys.exit()
    P.cell = cell

# TODO: Redesign time census so that only one is relevant at each source loop
def set_census_time_idx(P):
    t = P.time
    idx = binary_search(t, mcdc.settings.census_time) + 1

    if idx == len(mcdc.settings.census_time):
        P.alive = False
        idx = None
    elif P.time == mcdc.settings.census_time[idx]:
        idx += 1
    P.idx_census_time = idx

#==============================================================================
# Move to event
#==============================================================================

def move_to_event(P):
    # Get speed
    P.speed = P.cell.material.speed[P.group]

    # Get distances to events
    d_collision        = collision_distance(P)
    surface, d_surface = surface_distance(P)
    d_mesh             = mcdc.tally.mesh.distance(P)
    d_census           = census_distance(P)

    # Determine event
    event, distance = determine_event(d_collision, d_surface, d_mesh, d_census)

    # Score tracklength tallies
    score_tracklength(P, distance)

    # Add a small-kick to ensure mesh crossing
    if event == EVENT_MESH:
        distance += PRECISION

    # Move particle
    move_particle(P, distance)

    # Record surface if crossed
    if event == EVENT_SURFACE:
        P.surface = surface

    return event

def collision_distance(P):
    # Get total XS
    SigmaT = P.cell.material.total[P.group]

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Time absorption?
    if mcdc.settings.mode_alpha:
        SigmaT += abs(mcdc.global_tally.alpha_eff)/P.speed

    # Sample collision distance
    xi     = mcdc.rng()
    distance  = -np.log(xi)/SigmaT
    return distance
    
def surface_distance(P):
    surface  = None
    distance = np.inf
    for S in P.cell.surfaces:
        speed = P.cell.material.speed[P.group]
        d = S.distance(P.position, P.direction, P.time, speed)
        if d < distance:
            surface  = S
            distance = d
    return surface, distance

def census_distance(P):
    t_census = mcdc.settings.census_time[P.idx_census_time]
    distance = P.speed*(t_census - P.time)
    return distance

def determine_event(d_collision, d_surface, d_mesh, d_census):
    event  = EVENT_COLLISION
    distance = d_collision
    if distance > d_surface:
        event  = EVENT_SURFACE
        distance = d_surface
    if distance > d_census:
        event  = EVENT_CENSUS
        distance = d_census
    if distance > d_mesh:
        event  = EVENT_MESH
        distance = d_mesh
    return event, distance

def score_tracklength(P, distance):
    mcdc.tally.score(P, distance)

    # Score eigenvalue tallies
    if mcdc.settings.mode_eigenvalue:
        nu       = P.cell.material.nu_p[P.group]\
                   + sum(P.cell.material.nu_d[P.group])
        SigmaF   = P.cell.material.fission[P.group]
        nuSigmaF = nu*SigmaF
        mcdc.global_tally.nuSigmaF_sum += P.weight*distance*nuSigmaF

        if mcdc.settings.mode_alpha:
            mcdc.global_tally.ispeed_sum += P.weight*distance\
                                       /P.cell.material.speed[P.group]

def move_particle(P, distance):
    P.position.x += P.direction.x*distance
    P.position.y += P.direction.y*distance
    P.position.z += P.direction.z*distance
    P.time       += distance/P.speed


#==============================================================================
# Surface crossing
#==============================================================================

def surface_crossing(P):
    # Implement BC
    P.surface.bc(P)

    # Small kick to make sure crossing
    move_particle(P, PRECISION)
 
    # Set new cell
    if P.alive:
        set_cell(P)

#==============================================================================
# Collision
#==============================================================================

def collision(P):
    SigmaT = P.cell.material.total[P.group]
    SigmaC = P.cell.material.capture[P.group]
    SigmaS = P.cell.material.scatter[P.group]
    SigmaF = P.cell.material.fission[P.group]

    if mcdc.settings.mode_alpha:
        Sigma_alpha = abs(mcdc.global_tally.alpha_eff)/P.speed
        SigmaT += Sigma_alpha

    if mcdc.settings.implicit_capture:
        capture = SigmaC
        if mcdc.settings.mode_alpha:
            capture += Sigma_alpha
        P.weight *= (SigmaT-capture)/SigmaT
        SigmaT -= capture

    # Sample collision type
    xi = mcdc.rng()*SigmaT
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

def capture(P):
    P.alive = False

#==============================================================================
# Scattering
#==============================================================================

def scattering(P):
    # Ger outgoing spectrum
    chi_s = P.cell.material.chi_s[P.group]
    G     = P.cell.material.G
    
    # Sample outgoing energy
    xi  = mcdc.rng()
    tot = 0.0
    for g_out in range(G):
        tot += chi_s[g_out]
        if tot > xi:
            break
    P.group = g_out
    
    # Sample scattering angle
    mu = 2.0*mcdc.rng() - 1.0;
    
    # Sample azimuthal direction
    azi     = 2.0*np.pi*mcdc.rng()
    cos_azi = np.cos(azi)
    sin_azi = np.sin(azi)
    Ac      = (1.0 - mu**2)**0.5

    ux = P.direction.x
    uy = P.direction.y
    uz = P.direction.z
    
    if uz != 1.0:
        B = (1.0 - P.direction.z**2)**0.5
        C = Ac/B
        
        P.direction.x = ux*mu + (ux*uz*cos_azi - uy*sin_azi)*C
        P.direction.y = uy*mu + (uy*uz*cos_azi + ux*sin_azi)*C
        P.direction.z = uz*mu - cos_azi*Ac*B
    
    # If dir = 0i + 0j + k, interchange z and y in the scattering formula
    else:
        B = (1.0 - uy**2)**0.5
        C = Ac/B
        
        P.direction.x = ux*mu + (ux*uy*cos_azi - uz*sin_azi)*C
        P.direction.z = uz*mu + (uz*uy*cos_azi + ux*sin_azi)*C
        P.direction.y = uy*mu - cos_azi*Ac*B
        
#==============================================================================
# Fission
#==============================================================================

def fission(P):
    # Kill the current particle
    P.alive = False

    # Get group numbers
    G = P.cell.material.G
    J = P.cell.material.J

    # Total nu
    nu_p = P.cell.material.nu_p[P.group]
    nu = nu_p
    if J>0: 
        nu_d = P.cell.material.nu_d[P.group]
        nu += sum(nu_d)

    # Sample number of fission neutrons
    #   in fixed-source, k_eff = 1.0
    N = floor(P.weight*nu/mcdc.global_tally.k_eff + mcdc.rng())

    # Push fission neutrons to bank
    for n in range(N):
        # Create copy
        P_new = P.copy()
        P_new.weight = 1.0

        # Determine if it's prompt or delayed neutrons, 
        # then get the energy spectrum and decay constant
        xi  = mcdc.rng()*nu
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
            xi = mcdc.rng()
            P_new.time = P.time - np.log(xi)/decay

            # Skip if it's beyond final census time
            if P_new.time > mcdc.settings.census_time[-1]:
                continue
            else:
                set_census_time_idx(P_new)

        # Sample outgoing energy
        xi  = mcdc.rng()
        tot = 0.0
        for g_out in range(G):
            tot += spectrum[g_out]
            if tot > xi:
                break
        P_new.group = g_out

        # Sample isotropic direction
        P_new.direction = isotropic_direction()

        # Bank
        mcdc.bank_fission.append(P_new)

#==============================================================================
# Time reaction
#==============================================================================

def time_reaction(P):
    if mcdc.global_tally.alpha_eff > 0:
        P.alive = False
    else:
        P_new = Particle.copy()
        mcdc.bank_history.append(P_new)

#==============================================================================
# Time census
#==============================================================================

def event_census(P):
    # Cross the time boundary
    d = PRECISION*P.speed
    move_particle(P, d)

    # Increment index
    P.idx_census_time += 1
    # Not final census?
    if P.idx_census_time < len(mcdc.settings.census_time):
        # Store for next time census
        bank_stored.append(P.copy())
    P.alive = False

#==============================================================================
# Miscellany
#==============================================================================

def binary_search(val, grid):
    """
    Binary search that returns the bin index of a value `val` given
    grid `grid`
    
    Some special cases:
        `val` < min(`grid`)  --> -1
        `val` > max(`grid`)  --> size of bins
        `val` = a grid point --> bin location whose upper bound is `val`
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
