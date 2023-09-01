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

do_nothing_id = 0


def generate_do_nothing(arg_count):
    global do_nothing_id
    name = f"do_nothing_{do_nothing_id}"
    args = ", ".join([f"arg_{i}" for i in range(arg_count)])
    source  =  "@njit\n"
    source += f"def {name}({args}):\n"
    source +=  "    pass\n"
    print(source)
    exec(source)
    result = eval(name)
    do_nothing_id += 1
    return result


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




def eval_toggle():
    global toggle_rosters
    for _, pair in toggle_rosters.items():
        val    = pair[0]
        roster = pair[1]
        for func in roster:
            if val:
                overwrite_func(func,numba.njit(func))
            else:
                arg_count = len(inspect.signature(func).parameters)
                overwrite_func(func,generate_do_nothing(arg_count))


def for_(target,on_target=[]):
    def for_inner(func):
        global target_rosters
        if target not in target_rosters:
            target_rosters[target] = {}
        target_rosters[target][func] = on_target
        return func
    return for_inner



def for_cpu(on_target =[]):
    return for_('cpu',on_target=on_target)

def for_gpu(on_target=[]):
    return for_('gpu',on_target=on_target)

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
# GPU Type / Extern Functions Forward Declarations
# =============================================================================


none_type = None
mcdc_type = None
state_spec = None
device_gpu = None
group_gpu = None
thread_gpu = None
particle_gpu = None
prep_gpu = None
step_gpu = None



def gpu_forward_declare():

    global none_type, mcdc_type, state_spec
    global device_gpu, group_gpu, thread_gpu
    global particle_gpu, particle_record_gpu
    global step_gpu

    none_type = numba.from_dtype(np.dtype([ ]))
    mcdc_type = numba.from_dtype(type_.global_)
    state_spec = (mcdc_type,none_type,none_type)
    device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
    particle_gpu = numba.from_dtype(type_.particle)
    particle_record_gpu = numba.from_dtype(type_.particle_record)

    def step(prog: numba.uintp, P: particle_gpu):
        pass

    step_gpu, =  adapt.harm.RuntimeSpec.async_dispatch(step)


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
    particle["fresh"] = True
    step_gpu(prog,particle)


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
def local_group_array():
    return np.zeros(1, dtype=type_.group_array)[0]
    
@for_gpu()
@cuda.jit
def local_group_array():
    return cuda.local.array(1, type_.group_array)[0]


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


@njit
def empty_base_func(prog):
    pass

def make_gpu_loop(state_spec,work_make_fn,step_fn,check_fn,arg_type,initial_fn=empty_base_func,final_fn=empty_base_func):
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

