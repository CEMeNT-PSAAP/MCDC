import inspect
import mcdc.config as config
import mcdc.adapt as adapt
import mcdc.type_ as type_
import numba
import ctypes
import time
import subprocess
import os
from mpi4py import MPI
from llvmlite import binding
import numpy as np


CACH_PATH = './__trace_cache__'

time_code = """
#include <iostream>
#include <chrono>
#include <cstdint>
extern "C"
int64_t mono_clock() {
    auto now = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return dur;
}
"""



mono_clock = None

@numba.njit()
def extern_gpu_clock_rate ():
    return 1000000000

if config.trace:
    if not os.path.exists(CACH_PATH):
        os.makedirs(CACH_PATH)
    base_path = f"{CACH_PATH}/trace"
    code_path = f"{base_path}.cpp"
    lib_path  = f"{base_path}.so"
    file = open(code_path,"w")
    file.write(time_code)
    file.close()
    cmd = f"g++ {code_path} --shared -fPIC -o {lib_path}"
    subprocess.run(cmd.split(),shell=False,check=True)
    abs_lib_path = os.path.abspath(lib_path)
    binding.load_library_permanently(abs_lib_path)
    sig = numba.types.int64()
    mono_clock = numba.types.ExternalFunction("mono_clock", sig)
    extern_gpu_clock_rate = numba.types.ExternalFunction("wall_clock_rate", sig)


@numba.njit()
def gpu_clock_rate():
    return extern_gpu_clock_rate()



trace_roster = {}

trace_wrapper_template = """
def trace_{id}_{name} ({arg_str}) :
    {trace_state_extractor}
    t0 = trace_get_clock()
    result = func ({arg_str})
    t1 = trace_get_clock()
    platform_index = trace_platform_index()
    adapt.global_add(trace['slots'][{id}]['runtime_total'],platform_index, t1 - t0)
    adapt.global_add(trace['slots'][{id}]['call_total'],platform_index,1)
    return result
"""

trace_wrapper_name_template = "trace_{name}"



sig     = numba.core.typing.signature
ext_fn  = numba.types.ExternalFunction
gpu_get_wall_clock = ext_fn("get_wall_clock",sig(numba.types.int64))


def get_clock():
    return time.monotonic_ns()

@numba.core.extending.overload(get_clock, target="cpu")
def cpu_get_clock():
    def inner_get_clock():
        return mono_clock()
    return inner_get_clock

@numba.core.extending.overload(get_clock, target="gpu")
def gpu_get_clock():
    def inner_get_clock():
        return gpu_get_wall_clock()
    return inner_get_clock


def platform_index():
    return 0

@numba.core.extending.overload(platform_index, target="cpu")
def cpu_platform_index():
    def inner_platform_index():
        return 1
    return inner_platform_index

@numba.core.extending.overload(platform_index, target="gpu")
def gpu_platform_index():
    def inner_platform_index():
        return 2
    return inner_platform_index


def trace(transforms=[]):

    def trace_inner(func):
        global get_clock
        global platform_index
        trace_get_clock = get_clock
        trace_platform_index = platform_index

        name = func.__name__
        arg_set = inspect.signature(func).parameters

        trace_state_extractors = {
            "mcdc": "trace = mcdc['trace']",
            "prog": "trace = adapt.mcdc_global(prog)['trace']",
            "mcdc_arr": "trace = mcdc_arr[0]['trace']",
        }

        extractor_target = None
        for target, extractor in trace_state_extractors.items():
            if target in arg_set:
                extractor_target = target
                break

        if config.trace and (extractor_target != None):

            global trace_roster
            import mcdc.adapt as adapt

            for tr in transforms:
                func = tr(func)

            if func not in trace_roster:
                trace_roster[name] = {'id': len(trace_roster)}

            func_id = trace_roster[name]['id']
            arg_str = ",".join([arg for arg in arg_set])

            trace_wrapper_source = trace_wrapper_template.format(
                name=name,
                arg_str=arg_str,
                id=func_id,
                trace_state_extractor=trace_state_extractors[extractor_target]
            )
            exec(trace_wrapper_source,locals(),locals())
            trace_func = eval(f"trace_{func_id}_{name}")
            return trace_func
        else:
            return func

    return trace_inner



def njit(*args,**kwargs):

    def trace_njit_inner(func):
        trace_func = trace(transforms=[numba.njit(*args,**kwargs)])(func)
        if (trace_func == func):
            return numba.njit(*args,**kwargs)(func)
        else:
            return numba.njit()(trace_func)

    return trace_njit_inner





def output_report(mcdc):

    report = open("report.csv","w")
    report.write("function name, ")
    report.write("python total runtime (ns), python total calls, ")
    report.write("cpu total runtime (ns), cpu total calls, ")
    report.write("gpu total runtime (mystery units), gpu total calls, ")
    reprot.write("\n")

    gpu_rate = 1000000000
    if config.target == "gpu":
        gpu_rate = gpu_clock_rate()

    multi_rank = True

    for name, info in trace_roster.items():
        func_id = info['id']
        slot = mcdc['trace']['slots'][func_id]

        if multi_rank:
            slot_arr = np.empty((1,),type_.trace_slot)
            MPI.COMM_WORLD.Allreduce(slot['runtime_total'],slot_arr[0]['runtime_total'])
            MPI.COMM_WORLD.Allreduce(slot['call_total'],slot_arr[0]['call_total'])
            slot['runtime_total'] = slot_arr[0]['runtime_total']
            slot['call_total'] = slot_arr[0]['call_total']

        python_nsecs = slot['runtime_total'][0]
        python_calls = slot['call_total'][0]
        cpu_nsecs = slot['runtime_total'][1]
        cpu_calls = slot['call_total'][1]
        gpu_nsecs = slot['runtime_total'][2] * 1000000000.0 / gpu_rate
        gpu_calls = slot['call_total'][2]
        report.write(f"{name},")
        report.write(f"{python_nsecs},{python_calls},")
        report.write(f"{cpu_nsecs},{cpu_calls},")
        report.write(f"{gpu_nsecs},{gpu_calls},")
        report.write("\n")
    report.close()

