import numpy as np
from numba import njit, objmode, literal_unroll, cuda
import numba
import mcdc.type_  as type_
import mcdc.kernel as kernel

path_to_harmonize='/home/brax/harmonize/code/'
import sys
sys.path.append(path_to_harmonize)
import harmonize as harm







# =============================================================================
# Seperate GPU/CPU Functions to Target Different Platforms
# =============================================================================

def device(prog):
    pass

@njit
def device_cpu(prog):
    return prog


def group(prog):
    pass

@njit
def group_cpu(prog):
    return prog


def thread(prog):
    pass

@njit
def thread_cpu(prog):
    return prog



def add_active(prog, particle):
    pass

@njit
def add_active_cpu(prog, particle):
    kernel.add_particle(kernel.copy_particle(particle), prog["bank_active"])

@cuda.jit
def add_active_gpu(prog, particle):
    pass



def add_source(prog, particle):
    pass

@njit
def add_source_cpu(prog, particle):
    kernel.add_particle(kernel.copy_particle(particle), prog["bank_source"])

@cuda.jit
def add_source_gpu(prog, particle):
    pass



def add_census(prog, particle):
    pass

@njit
def add_census_cpu(prog, particle):
    kernel.add_particle(kernel.copy_particle(particle), prog["bank_census"])

@cuda.jit
def add_census_gpu(prog, particle):
    pass



def add_IC(prog, particle):
    pass

@njit
def add_IC_cpu(prog, particle):
    kernel.add_particle(kernel.copy_particle(particle), prog["technique"]["IC_bank_neutron_local"])

@cuda.jit
def add_IC_gpu(prog, particle):
    pass





def local_particle():
    pass

@njit
def local_particle_cpu():
    return np.zeros(1, dtype=type_.particle)[0]

@cuda.jit
def local_particle_gpu():
    return cuda.local.array(1, dtype=type_.particle)[0]
    

def local_particle_record():
    pass

@njit
def local_particle_record_cpu():
    return np.zeros(1, dtype=type_.particle_record)[0]

@cuda.jit
def local_particle_record_gpu():
    return cuda.local.array(1, dtype=type_.particle_record)[0]


def global_add(ary,idx,val):
    pass

@njit
def global_add_cpu(ary,idx,val):
    result    = ary[idx]
    ary[idx] += val
    return result

@cuda.jit
def global_add_gpu(ary,idx,val):
    return cuda.atomic.add(ary,idx,val)

def global_max(ary,idx,val):
    pass

@njit
def global_max_cpu(ary,idx,val):
    result    = ary[idx]
    if ary[idx] < val :
        ary[idx] = val
    return result

@cuda.jit
def global_max_gpu(ary,idx,val):
    return cuda.atomic.max(ary,idx,val)



def make_utils(target):

    global global_add, global_add_cpu, global_add_gpu
    global global_max, global_max_cpu, global_max_gpu

    global local_particle, local_particle_cpu, local_particle_gpu
    global local_particle_record, local_particle_record_cpu, local_particle_record_gpu

    global device, device_cpu, device_gpu
    global group,  group_cpu,  group_gpu
    global thread, thread_cpu, thread_gpu

    global add_active, add_active_cpu, add_active_gpu
    global add_source, add_source_cpu, add_source_gpu
    global add_census, add_census_cpu, add_census_gpu
    
    if   target == 'cpu':
        global_add = global_add_cpu
        global_max = global_max_cpu
        local_particle = local_particle_cpu
        local_particle_record = local_particle_record_cpu
        device = device_cpu
        group  = group_cpu
        thread = thread_cpu
        add_active = add_active_cpu
        add_source = add_source_cpu
        add_census = add_census_cpu
    elif target == 'gpu':
        global_add = global_add_gpu
        global_max = global_max_gpu
        local_particle_record = local_particle_record_gpu
        device = device_gpu
        group  = group_gpu
        thread = thread_gpu
        add_active = add_active_gpu
        add_source = add_source_gpu
        add_census = add_census_gpu
    else:
        print(f"ERROR: Unrecognized target '{target}'")



# =========================================================================
# Types
# =========================================================================

particle = None

def make_types(target):
    global particle
    if target == 'gpu':
        particle = numba.from_dtype(type_.particle)
    elif target != 'cpu':
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
    if target == 'gpu':
        dev_state_type = numba.from_dtype(type_.global_)
        grp_state_type = numba.from_dtype(np.dtype([ ]))
        thd_state_type = numba.from_dtype(np.dtype([ ]))
    elif target != 'cpu':
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

def make_work(prog: numba.uintp) -> numba.boolean:
    
    return False


# =========================================================================
# Other Async Functions
# =========================================================================

# Monolithic function combining all events
def iterate(prog: numba.uintp, particle):
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
        multi_event_fns = [source,move,scattering,fission,leakage,bcollision]
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




def adapt_to(target):
    make_utils(target)
    make_types(target)
    make_states(target)
    make_spec(target)



