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
import inspect

from mcdc.print_ import print_error

import mcdc.adapt as adapt


# =============================================================================
# Error Messangers
# =============================================================================

def unknown_target(target):
    print_error(f"ERROR: Unrecognized target '{target}'")


# =============================================================================
# Decorators
# =============================================================================

toggle_rosters = {}

target_rosters = {}

late_jit_roster = set()

def overwrite_func(func,revised_func):
    mod_name = func.__module__
    fn_name  = func.__name__
    module = __import__(mod_name,fromlist=[fn_name])
    setattr(module, fn_name, revised_func)



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



@njit
def do_nothing(*args):
    pass

def eval_toggle():
    global toggle_rosters
    for _, pair in toggle_rosters.items():
        val    = pair[0]
        roster = pair[1]
        for func in roster:
            if val:
                overwrite_func(func,numba.njit(func))
            else:
                overwrite_func(func,do_nothing)


def for_(target,on_target=[]):
    def for_inner(func):
        global target_rosters
        if target not in target_rosters:
            target_rosters[target] = {}
        target_rosters[target][func] = on_target
        return func
    return for_inner



def for_cpu(on_target =[]):
    return for_('cpu',on_target)

def for_gpu(on_target=[]):
    return for_('gpu',on_target)

def target_for(target):

    for func in late_jit_roster:
        if target == 'cpu':
            overwrite_func(func,numba.njit)
        elif target == 'gpu':
            overwrite_func(func,numba.cuda.jit)
        else:
            unknown_target(target)

    for func, on_target in target_rosters[target].items():
        transformed = func
        for transformer in on_target:
            transformed = transformer(transformed)
        name = func.__name__
        print(f"Overwriting func {name} for target {target}")
        overwrite_func(func, transformed)

def jit_on_target():
    def jit_on_target_inner(func):
        late_jit_roster.add(func)
        return func
    return jit_on_target_inner




# Function adapted from Phillip Eller's `myjit` solution to the GPU/CPU array
# problem brought up in https://github.com/numba/numba/issues/2571
def universal_arrays(target):
    def universal_arrays_inner (func):
        if target == 'gpu':
            source = inspect.getsource(func).splitlines()
            for idx, line in enumerate(source):
                if '@universal_arrays' in line:
                    source = '\n'.join(source[idx+1:]) + '\n'
                    break
            source = source.replace('np.empty','cuda.local.array')
            print(source)
            exec(source)
            revised_func = eval(func.__name__)
            overwrite_func(func,revised_func)
            #module = __import__(func.__module__,fromlist=[func.__name__])
            #print(inspect.getsource(getattr(module,func.__name__)))
        return revised_func
    return universal_arrays_inner




# =============================================================================
# Seperate GPU/CPU Functions to Target Different Platforms
# =============================================================================

@for_cpu()
@njit
def device(prog):
    return prog

@for_gpu()
@cuda.jit
def device(prog):
    return device_gpu(prog)


@for_cpu()
@njit
def group(prog):
    return prog

@for_gpu()
@cuda.jit
def group(prog):
    return group_gpu(prog)


@for_cpu()
@njit
def thread(prog):
    return prog

@for_gpu()
@cuda.jit
def thread(prog):
    return thread_gpu(prog)




@for_cpu()
@njit
def add_active(particle,prog):
    kernel.add_particle(particle, prog["bank_active"])

@for_gpu()
@cuda.jit
def add_active(particle,prog):
    #iterate_async(prog,particle)
    kernel.add_particle(particle, prog["bank_active"])


@for_cpu()
@njit
def add_source(particle, prog):
    kernel.add_particle(particle, prog["bank_source"])

@for_gpu()
@cuda.jit
def add_source(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["bank_source"])



@for_cpu()
@njit
def add_census(particle, prog):
    kernel.add_particle(particle, prog["bank_census"])

@for_gpu()
@cuda.jit
def add_census(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["bank_census"])




@for_cpu()
@njit
def add_IC(particle, prog):
    kernel.add_particle(particle, prog["technique"]["IC_bank_neutron_local"])

@for_gpu()
@cuda.jit
def add_IC(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["technique"]["IC_bank_neutron_local"])




@for_cpu()
@njit
def local_translate():
    return np.zeros(1, dtype=type_.translate)[0]
    
@for_gpu()
@cuda.jit
def local_translate():
    trans = cuda.local.array(1, type_.translate)[0]
    for i in range(3):
        trans["values"][i] = 0
    return trans



@for_cpu()
@njit
def local_particle():
    return np.zeros(1, dtype=type_.particle)[0]

@for_gpu()
@cuda.jit
def local_particle():
    return cuda.local.array(1, dtype=type_.particle)[0]


@for_cpu()
@njit
def local_particle_record():
    return np.zeros(1, dtype=type_.particle_record)[0]

@for_gpu()
@cuda.jit
def local_particle_record():
    return cuda.local.array(1, dtype=type_.particle_record)[0]


@for_cpu()
@njit
def global_add(ary,idx,val):
    result    = ary[idx]
    ary[idx] += val
    return result

@for_gpu()
@cuda.jit
def global_add(ary,idx,val):
    return cuda.atomic.add(ary,idx,val)




@for_cpu()
@njit
def global_max(ary,idx,val):
    result    = ary[idx]
    if ary[idx] < val :
        ary[idx] = val
    return result

@for_gpu()
@cuda.jit
def global_max(ary,idx,val):
    return cuda.atomic.max(ary,idx,val)



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
        unknown_target(target)



def make_gpu_loop(state_spec,work_make_fn,step_fn,check_fn,arg_type,initial_fn=do_nothing,final_fn=do_nothing):
    async_fn_list = [step_fn]
    device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
    def make_work(prog: numba.uintp) -> numba.boolean:
        return work_make_fn(prog)
    def initialize(prog: numba.uintp):
        initial_fn(prog)
    def finalize(prog: numba.uintp):
        final_fn(prog)
    def step(prog: numba.uintp, arg: arg_type):
        
        step_async()

    step_async, = harm.RuntimeSpec.async_dispatch(step)

    pass


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
        unknown_target(target)


@njit
def local_array(a,b):
    pass

def manual_target(target):
    global local_array
    if   target == 'cpu':
        local_array = np.empty
    elif target == 'gpu':
        local_array = numba.cuda.jit(numba.cuda.local.array)
    else:
        unknown_target(target)




