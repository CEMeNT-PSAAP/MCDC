
import numpy as np
import numba

import mcdc.type_ as type_
import mcdc.adapt as adapt
import mcdc.trace as trace

if adapt.HAS_HARMONIZE:
    import harmonize as harm


# =============================================================================
# GPU Type / Extern Functions Forward Declarations
# =============================================================================


SIMPLE_ASYNC = True

none_type = None
mcdc_global_type = None
mcdc_data_type = None
state_spec = None
mcdc_global_gpu = None
mcdc_data_gpu = None
group_gpu = None
thread_gpu = None
particle_gpu = None
prep_gpu = None
step_async = None
halt_early = None
find_cell_async = None


def gpu_forward_declare(args):

    if args.gpu_rocm_path != None:
        harm.config.set_rocm_path(args.gpu_rocm_path)

    if args.gpu_cuda_path != None:
        harm.config.set_cuda_path(args.gpu_cuda_path)

    global none_type, mcdc_global_type, mcdc_data_type
    global state_spec
    global mcdc_global_gpu, mcdc_data_gpu
    global group_gpu, thread_gpu
    global particle_gpu, particle_record_gpu
    global step_async, find_cell_async, halt_early

    none_type = numba.from_dtype(np.dtype([]))
    mcdc_global_type = numba.from_dtype(type_.global_)
    mcdc_data_type = numba.from_dtype(type_.tally)
    state_spec = (
        {
            "global": mcdc_global_type,
            "data": mcdc_data_type,
        },
        none_type,
        none_type,
    )
    access_fns = harm.RuntimeSpec.access_fns(state_spec)
    mcdc_global_gpu = access_fns["device"]["global"]
    mcdc_data_gpu = access_fns["device"]["data"]
    group_gpu = access_fns["group"]
    thread_gpu = access_fns["thread"]
    particle_gpu = numba.from_dtype(type_.particle)
    particle_record_gpu = numba.from_dtype(type_.particle_record)

    def step(prog: numba.uintp, P: particle_gpu):
        pass

    def find_cell(prog: numba.uintp, P: particle_gpu):
        pass

    step_async, find_cell_async = adapt.harm.RuntimeSpec.async_dispatch(step, find_cell)
    interface = adapt.harm.RuntimeSpec.program_interface()
    halt_early = interface["halt_early"]


# =============================================================================
# Seperate GPU/CPU Functions to Target Different Platforms
# =============================================================================


def mcdc_global(prog):
    return prog


@adapt.for_cpu()
def mcdc_global(prog):
    return prog


@adapt.for_gpu()
def mcdc_global(prog):
    return mcdc_global_gpu(prog)


@adapt.for_cpu()
def mcdc_data(prog):
    return None


@adapt.for_gpu()
def mcdc_data(prog):
    return mcdc_data_gpu(prog)


@adapt.for_cpu()
def group(prog):
    return prog


@adapt.for_gpu()
def group(prog):
    return group_gpu(prog)


@adapt.for_cpu()
def thread(prog):
    return prog


@adapt.for_gpu()
def thread(prog):
    return thread_gpu(prog)



@adapt.for_cpu()
def global_add(ary, idx, val):
    result = ary[idx]
    ary[idx] += val
    return result


@adapt.for_gpu()
def global_add(ary, idx, val):
    return harm.array_atomic_add(ary, idx, val)


@adapt.for_cpu()
def global_max(ary, idx, val):
    result = ary[idx]
    if ary[idx] < val:
        ary[idx] = val
    return result


@adapt.for_gpu()
def global_max(ary, idx, val):
    return harm.array_atomic_max(ary, idx, val)


# =========================================================================
# Program Specifications
# =========================================================================

state_spec = None
one_event_fns = None
multi_event_fns = None


device_gpu, group_gpu, thread_gpu = None, None, None
iterate_async = None


def make_spec(target):
    global state_spec, one_event_fns, multi_event_fns
    global device_gpu, group_gpu, thread_gpu
    global iterate_async
    if target == "gpu":
        state_spec = (dev_state_type, grp_state_type, thd_state_type)
        one_event_fns = [iterate]
        # multi_event_fns = [source,move,scattering,fission,leakage,bcollision]
        device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
        (iterate_async,) = harm.RuntimeSpec.async_dispatch(iterate)
    elif target != "cpu":
        unknown_target(target)


@numba.njit
def empty_base_func(prog):
    pass


def make_gpu_loop(
    state_spec,
    work_make_fn,
    step_fn,
    check_fn,
    arg_type,
    initial_fn=empty_base_func,
    final_fn=empty_base_func,
):
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

    (step_async,) = harm.RuntimeSpec.async_dispatch(step)

    pass


# =========================================================================
# Compilation and Main Adapter
# =========================================================================


def compiler(func, target):
    if target == "cpu":
        return jit(func, nopython=True, nogil=True)  # , parallel=True)
    elif target == "cpus":
        return jit(func, nopython=True, nogil=True, parallel=True)
    elif target == "gpu_device":
        return cuda.jit(func, device=True)
    elif target == "gpu":
        return cuda.jit(func)
    else:
        unknown_target(target)


