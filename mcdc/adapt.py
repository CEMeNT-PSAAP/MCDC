import numpy as np
from numba import njit, jit, objmode, literal_unroll, cuda, types
from numba.extending import intrinsic
import numba
import mcdc.type_ as type_
import mcdc.kernel as kernel
import mcdc.loop as loop

try:
    import harmonize as harm

    HAS_HARMONIZE = True
except:
    HAS_HARMONIZE = False

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
# uintp/voidptr casters
# =============================================================================


@intrinsic
def cast_uintp_to_voidptr(typingctx, src):
    # check for accepted types
    if isinstance(src, types.Integer):
        # create the expected type signature
        result_type = types.voidptr
        sig = result_type(types.uintp)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.inttoptr(src, llrtype)

        return sig, codegen


@intrinsic
def cast_voidptr_to_uintp(typingctx, src):
    # check for accepted types
    if isinstance(src, types.RawPointer):
        # create the expected type signature
        result_type = types.uintp
        sig = result_type(types.voidptr)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.ptrtoint(src, llrtype)

        return sig, codegen


# =============================================================================
# Decorators
# =============================================================================

toggle_rosters = {}

target_rosters = {}

late_jit_roster = set()

do_nothing_id = 0


def generate_do_nothing(arg_count, crash_on_call=None):
    global do_nothing_id
    name = f"do_nothing_{do_nothing_id}"
    args = ", ".join([f"arg_{i}" for i in range(arg_count)])
    source = f"def {name}({args}):\n"
    if crash_on_call != None:
        source += f"    assert False, '{crash_on_call}'\n"
    else:
        source += "    pass\n"
    exec(source)
    result = eval(name)
    do_nothing_id += 1
    return result


def overwrite_func(func, revised_func):
    mod_name = func.__module__
    fn_name = func.__name__
    new_fn_name = revised_func.__name__
    # print(f"Overwriting function {fn_name} in module {mod_name} with {new_fn_name}")
    module = __import__(mod_name, fromlist=[fn_name])
    setattr(module, fn_name, revised_func)


def toggle(flag):
    def toggle_inner(func):
        global toggle_rosters
        if flag not in toggle_rosters:
            toggle_rosters[flag] = [False, []]
        toggle_rosters[flag][1].append(func)
        return func

    return toggle_inner


def set_toggle(flag, val):
    toggle_rosters[flag][0] = val


def eval_toggle():
    global toggle_rosters
    for _, pair in toggle_rosters.items():
        val = pair[0]
        roster = pair[1]
        for func in roster:
            if val:
                overwrite_func(func, numba.njit(func))
            else:
                global do_nothing_id
                name = func.__name__
                # print(f"do_nothing_{do_nothing_id} for {name}")
                arg_count = len(inspect.signature(func).parameters)
                overwrite_func(func, numba.njit(generate_do_nothing(arg_count)))


blankout_roster = {}


def blankout_fn(func):
    global blankout_roster

    mod_name = func.__module__
    fn_name = func.__name__
    id = (mod_name, fn_name)

    if id not in blankout_roster:
        global do_nothing_id
        name = func.__name__
        # print(f"do_nothing_{do_nothing_id} for {name}")
        arg_count = len(inspect.signature(func).parameters)
        blankout_roster[id] = generate_do_nothing(
            arg_count, crash_on_call=f"blankout fn for {name} should never be called"
        )

    blank = blankout_roster[id]

    return blank


def for_(target, on_target=[]):
    # print(f"{target}")
    def for_inner(func):
        global target_rosters
        mod_name = func.__module__
        fn_name = func.__name__
        # print(f"{target} {mod_name} {fn_name}")
        params = inspect.signature(func).parameters
        if target not in target_rosters:
            target_rosters[target] = {}
        target_rosters[target][(mod_name, fn_name)] = func
        # blank = blankout_fn(func)
        param_str = ", ".join(p for p in params)
        jit_str = f"def jit_func({param_str}):\n    global target_rosters\n    return target_rosters['{target}'][('{mod_name}','{fn_name}')]"
        # print(jit_str)
        exec(jit_str, globals(), locals())
        result = eval("jit_func")
        blank = blankout_fn(func)
        numba.core.extending.overload(blank, target=target)(result)
        return blank

    return for_inner


def for_cpu(on_target=[]):
    return for_("cpu", on_target=on_target)


def for_gpu(on_target=[]):
    return for_("gpu", on_target=on_target)


def target_for(target):
    pass


#
#    for func in late_jit_roster:
#        if target == 'cpu':
#            overwrite_func(func,numba.njit)
#        elif target == 'gpu':
#            overwrite_func(func,numba.cuda.jit)
#        else:
#            unknown_target(target)
#
#    for func, on_target in target_rosters[target].items():
#        transformed = func
#        for transformer in on_target:
#            transformed = transformer(transformed)
#        name = func.__name__
#        print(f"Overwriting func {name} for target {target}")
#        overwrite_func(func, transformed)


def jit_on_target():
    def jit_on_target_inner(func):
        late_jit_roster.add(func)
        return func

    return jit_on_target_inner


def nopython_mode(is_on):
    if is_on:
        return
    if not isinstance(target_rosters["cpu"], dict):
        return

    for impl in target_rosters["cpu"].values():
        overwrite_func(impl, impl)


# Function adapted from Phillip Eller's `myjit` solution to the GPU/CPU array
# problem brought up in https://github.com/numba/numba/issues/2571
def universal_arrays(target):
    def universal_arrays_inner(func):
        if target == "gpu":
            source = inspect.getsource(func).splitlines()
            for idx, line in enumerate(source):
                if "@universal_arrays" in line:
                    source = "\n".join(source[idx + 1 :]) + "\n"
                    break
            source = source.replace("np.empty", "cuda.local.array")
            # print(source)
            exec(source)
            revised_func = eval(func.__name__)
            overwrite_func(func, revised_func)
            # module = __import__(func.__module__,fromlist=[func.__name__])
            # print(inspect.getsource(getattr(module,func.__name__)))
        return revised_func

    return universal_arrays_inner


# =============================================================================
# GPU Type / Extern Functions Forward Declarations
# =============================================================================


SIMPLE_ASYNC = True

none_type = None
mcdc_type = None
state_spec = None
device_gpu = None
group_gpu = None
thread_gpu = None
particle_gpu = None
prep_gpu = None
step_async = None
find_cell_async = None


def gpu_forward_declare():

    global none_type, mcdc_type, state_spec
    global device_gpu, group_gpu, thread_gpu
    global particle_gpu, particle_record_gpu
    global step_async, find_cell_async

    none_type = numba.from_dtype(np.dtype([]))
    mcdc_type = numba.from_dtype(type_.global_)
    state_spec = (mcdc_type, none_type, none_type)
    device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
    particle_gpu = numba.from_dtype(type_.particle)
    particle_record_gpu = numba.from_dtype(type_.particle_record)

    def step(prog: numba.uintp, P: particle_gpu):
        pass

    def find_cell(prog: numba.uintp, P: particle_gpu):
        pass

    step_async, find_cell_async = adapt.harm.RuntimeSpec.async_dispatch(step, find_cell)


# =============================================================================
# Seperate GPU/CPU Functions to Target Different Platforms
# =============================================================================


@for_cpu()
def device(prog):
    return prog


@for_gpu()
def device(prog):
    return device_gpu(prog)


@for_cpu()
def group(prog):
    return prog


@for_gpu()
def group(prog):
    return group_gpu(prog)


@for_cpu()
def thread(prog):
    return prog


@for_gpu()
def thread(prog):
    return thread_gpu(prog)


@for_cpu()
def add_active(particle, prog):
    kernel.add_particle(particle, prog["bank_active"])


@for_gpu()
def add_active(particle, prog):
    P = kernel.recordlike_to_particle(particle)
    if SIMPLE_ASYNC:
        step_async(prog, P)
    else:
        find_cell_async(prog, P)


@for_cpu()
def add_source(particle, prog):
    kernel.add_particle(particle, prog["bank_source"])


@for_gpu()
def add_source(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["bank_source"])


@for_cpu()
def add_census(particle, prog):
    kernel.add_particle(particle, prog["bank_census"])


@for_gpu()
def add_census(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["bank_census"])


@for_cpu()
def add_IC(particle, prog):
    kernel.add_particle(particle, prog["technique"]["IC_bank_neutron_local"])


@for_gpu()
def add_IC(particle, prog):
    mcdc = device(prog)
    kernel.add_particle(particle, mcdc["technique"]["IC_bank_neutron_local"])


@for_cpu()
def local_translate():
    return np.zeros(1, dtype=type_.translate)[0]


@for_gpu()
def local_translate():
    trans = cuda.local.array(1, type_.translate)[0]
    for i in range(3):
        trans["values"][i] = 0
    return trans


@for_cpu()
def local_group_array():
    return np.zeros(1, dtype=type_.group_array)[0]


@for_gpu()
def local_group_array():
    return cuda.local.array(1, type_.group_array)[0]


@for_cpu()
def local_j_array():
    return np.zeros(1, dtype=type_.j_array)[0]


@for_gpu()
def local_j_array():
    return cuda.local.array(1, type_.j_array)[0]


@for_cpu()
def local_particle():
    return np.zeros(1, dtype=type_.particle)[0]


@for_gpu()
def local_particle():
    return cuda.local.array(1, dtype=type_.particle)[0]


@for_cpu()
def local_particle_record():
    return np.zeros(1, dtype=type_.particle_record)[0]


@for_gpu()
def local_particle_record():
    return cuda.local.array(1, dtype=type_.particle_record)[0]


@for_cpu()
def global_add(ary, idx, val):
    result = ary[idx]
    ary[idx] += val
    return result


@for_gpu()
def global_add(ary, idx, val):
    return cuda.atomic.add(ary, idx, val)


@for_cpu()
def global_max(ary, idx, val):
    result = ary[idx]
    if ary[idx] < val:
        ary[idx] = val
    return result


@for_gpu()
def global_max(ary, idx, val):
    return cuda.atomic.max(ary, idx, val)


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


@njit
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
