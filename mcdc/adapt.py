import numpy as np
from numba import njit, jit, objmode, literal_unroll, cuda
import numba
import mcdc.type_  as type_
import mcdc.kernel as kernel
import mcdc.loop   as loop

path_to_harmonize='/home/brax/harmonize/code/'
import sys
sys.path.append(path_to_harmonize)
import harmonize as harm
import math



toggle_rosters = {}

target_rosters = {}

def toggle(flag):
    def toggle_inner(func):
        global toggle_rosters
        if flag not in toggle_rosters:
            toggle_rosters[flag] = [False,[]]
        toggle_rosters[flag][1].append(func)
        return func
    return toggle_inner

def set_toggle(flag,val):
    toggle_rosters[flag][0] = val

def eval_toggle():
    @njit
    def do_nothing(*args):
        pass
    global toggle_rosters
    for _, pair in toggle_rosters.items():
        val    = pair[0]
        roster = pair[1]
        if val:
            continue
        for func in roster:
            mod_name = func.__module__
            fn_name  = func.__name__
            module = __import__(mod_name,fromlist=[fn_name])
            setattr(module, fn_name, do_nothing)


def for_cpu(func):
    global target_rosters
    if 'cpu' not in target_rosters:
        target_rosters['cpu'] = set()
    target_rosters['cpu'].add(func)


def for_gpu(func):
    global target_rosters
    if 'gpu' not in target_rosters:
        target_rosters['gpu'] = set()
    target_rosters['gpu'].add(func)

def target_for(target):
    for func in target_rosters[target]:
        mod_name = func.__module__
        fn_name  = func.__name__
        module = __import__(mod_name,fromlist=[fn_name])
        setattr(module, fn_name, func)



# =============================================================================
# Seperate GPU/CPU Functions to Target Different Platforms
# =============================================================================

@for_cpu
@njit
def device(prog):
    return prog

@for_gpu
@cuda.jit
def device(prog):
    return device_gpu(prog)


@for_cpu
@njit
def group(prog):
    return prog

@for_gpu
@cuda.jit
def group(prog):
    return group_gpu(prog)


@for_cpu
@njit
def thread(prog):
    return prog

@for_gpu
@cuda.jit
def thread(prog):
    return thread_gpu(prog)




@for_cpu
@njit
def add_active(particle,prog):
    kernel.add_particle(particle, prog["bank_active"])

@for_gpu
@cuda.jit
def add_active(particle,prog):
    #iterate_async(prog,particle)
    kernel.add_particle(particle, prog["bank_active"])


@for_cpu
@njit
def add_source(particle, prog):
    kernel.add_particle(particle, prog["bank_source"])

@for_gpu
@cuda.jit
def add_source(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["bank_source"])



@for_cpu
@njit
def add_census_cpu(particle, prog):
    kernel.add_particle(particle, prog["bank_census"])

@for_gpu
@cuda.jit
def add_census_gpu(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["bank_census"])




@for_cpu
@njit
def add_IC_cpu(particle, prog):
    kernel.add_particle(particle, prog["technique"]["IC_bank_neutron_local"])

@for_gpu
@cuda.jit
def add_IC_gpu(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["technique"]["IC_bank_neutron_local"])






@for_cpu
@njit
def local_particle():
    return np.zeros(1, dtype=type_.particle)[0]

@for_gpu
@cuda.jit
def local_particle():
    return cuda.local.array(1, dtype=type_.particle)[0]
    


@for_cpu
@njit
def local_particle_record():
    return np.zeros(1, dtype=type_.particle_record)[0]

@for_gpu
@cuda.jit
def local_particle_record():
    return cuda.local.array(1, dtype=type_.particle_record)[0]


@for_cpu
@njit
def global_add(ary,idx,val):
    result    = ary[idx]
    ary[idx] += val
    return result

@for_gpu
@cuda.jit
def global_add(ary,idx,val):
    return cuda.atomic.add(ary,idx,val)




@for_cpu
@njit
def global_max_cpu(ary,idx,val):
    result    = ary[idx]
    if ary[idx] < val :
        ary[idx] = val
    return result

@for_gpu
@cuda.jit
def global_max_gpu(ary,idx,val):
    return cuda.atomic.max(ary,idx,val)



def make_utils(target):
    
    if   target == 'cpu':
        pass
    elif target == 'gpu':
        device = device_gpu
        group  = group_gpu
        thread = thread_gpu
    else:
        print(f"ERROR: Unrecognized target '{target}'")



# =========================================================================
# Types
# =========================================================================

particle = None

def make_types(target):
    global particle
    if   target == 'cpu':
        pass
    elif target == 'gpu':
        particle = numba.from_dtype(type_.particle)
    else:
        print(f"ERROR: Unrecognized target '{target}'")


# =========================================================================
# Program State Types
# =========================================================================



dev_state_type = None
grp_state_type = None
thd_state_type = None

def make_states(target):
    global dev_state_type
    global grp_state_type
    global thd_state_type
    if target == 'cpu':
        pass
    elif target == 'gpu':
        dev_state_type = numba.from_dtype(type_.global_)
        grp_state_type = numba.from_dtype(np.dtype([ ]))
        thd_state_type = numba.from_dtype(np.dtype([ ]))
    else:
        print(f"ERROR: Unrecognized target '{target}'")



# =========================================================================
# Primary Loop Logic
# =========================================================================



# =========================================================================
# Base Async Functions
# =========================================================================


def initialize(prog: numba.uintp):
    pass

def finalize(prog: numba.uintp):
    pass

@njit
def make_work_source(prog):
    mcdc = device(prog)

    idx_work = global_add(mcdc["mpi_work_iter"],1)

    if idx_work >= mcdc["mpi_work_size"]:
        return False
    
    loop.generate_source_particle(idx_work,mcdc["source_seed"],mcdc)
    loop.exhaust_active_bank(mcdc)

    return True


@njit
def make_work_source_precursor(prog):
    mcdc = device(prog)

    idx_work = global_add(mcdc["mpi_work_iter"],1)

    if idx_work >= mcdc["mpi_work_size_precursor"]:
        return False

    # Get precursor
    DNP = mcdc["bank_precursor"]["precursors"][idx_work]

    # Determine number of particles to be generated
    w = DNP["w"]
    N = math.floor(w)
    # "Roulette" the last particle
    seed_work = kernel.split_seed(idx_work, mcdc["source_precursor_seed"])
    if kernel.rng_from_seed(seed_work) < w - N:
        N += 1
    DNP["w"] = N

    for particle_idx in range(N):
        loop.generate_precursor_particle(DNP,particle_idx,seed_work,mcdc)

    return True


# =========================================================================
# Other Async Functions
# =========================================================================

# Monolithic function combining all events
def iterate(prog: numba.uintp, particle: type_.particle):
    pass


# =========================================================================
# Program Specifications
# =========================================================================

state_spec      = None 
one_event_fns   = None
multi_event_fns = None


device_gpu, group_gpu, thread_gpu = None, None, None
iterate_async   = None


def make_spec(target):
    global state_spec, one_event_fns, multi_event_fns
    global device_gpu, group_gpu, thread_gpu
    global iterate_async
    if target == 'gpu':
        state_spec = (dev_state_type,grp_state_type,thd_state_type) 
        one_event_fns   = [iterate]
        #multi_event_fns = [source,move,scattering,fission,leakage,bcollision]
        device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
        iterate_async, = harm.RuntimeSpec.async_dispatch(iterate)
    elif target != 'cpu':
        print(f"ERROR: Unrecognized target '{target}'")



# =========================================================================
# Program Instance Generation
# =========================================================================


def harmonize_factory():
    pass


# =========================================================================
# Main loop
# =========================================================================


def sim_particles(P,prog):
    pass

def sim_particle(P,prog):
    loop.loop_particle()

def particle_loop_gpu():
    pass


# =========================================================================
# Compilation and Main Adapter
# =========================================================================


def compiler(func, target):
    if target == 'cpu':
        return jit(func, nopython=True, nogil=True)#, parallel=True)
    elif target == 'cpus':
        return jit(func, nopython=True, nogil=True, parallel=True)
    elif target == 'gpu_device':
        return cuda.jit(func,device=True)
    elif target == 'gpu':
        return cuda.jit(func)
    else:
        print(f"[ERROR] Unrecognized target '{target}'.")



def make_loops(target):
    #pass]
    if target == 'cpu':
        loop.step_particle = compiler(loop.step_particle,target)
        loop.process_sources = compiler(loop.process_sources,target)
        loop.process_source_precursors = compiler(loop.process_source_precursors,target)
    else:
        assert False


def adapt_to(target):
    make_utils(target)
    make_types(target)
    make_states(target)
    make_spec(target)
    make_loops(target)



