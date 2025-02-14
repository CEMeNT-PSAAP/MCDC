import inspect
import mcdc.config as config
import mcdc.adapt as adapt
import numba
import ctypes
import time
import subprocess
from os.path import abspath
from llvmlite import binding



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

if config.trace:
    code_path = f"./trace.cpp"
    file = open(code_path,"w")
    file.write(time_code)
    file.close()
    cmd = "g++ trace.cpp --shared -fPIC -o ./trace.so"
    subprocess.run(cmd.split(),shell=False,check=True)
    so_path  = f"./trace.so"
    abs_so_path = abspath(so_path)
    binding.load_library_permanently(abs_so_path)
    sig = numba.types.int64()
    mono_clock = numba.types.ExternalFunction("mono_clock", sig)






trace_roster = {}

trace_wrapper_template = """
def trace_{id}_{name} ({arg_str}) :
    t0 = trace_get_clock()
    result = njit_func ({arg_str})
    t1 = trace_get_clock()
    adapt.global_add(mcdc['trace']['slots'][{id}]['runtime_total'],trace_platform_index(), t1 - t0)
    adapt.global_add(mcdc['trace']['slots'][{id}]['call_total'],trace_platform_index(),1)
    return result
"""

trace_wrapper_name_template = "trace_{name}"


@adapt.for_cpu()
def get_clock():
    return mono_clock()


sig     = numba.core.typing.signature
ext_fn  = numba.types.ExternalFunction
gpu_get_wall_clock = ext_fn("get_wall_clock",sig(numba.types.int64))

@adapt.for_gpu()
def get_clock():
    return gpu_get_wall_clock()


@adapt.for_cpu()
def platform_index():
    return 0

@adapt.for_gpu()
def platform_index():
    return 1


def njit(*args,**kwargs):

    def trace_njit_inner(func):
        trace_get_clock = get_clock
        trace_platform_index = platform_index

        name = func.__name__
        arg_set = inspect.signature(func).parameters

        if config.trace and ("mcdc" in arg_set):

            global trace_roster
            import mcdc.adapt as adapt

            njit_func = numba.njit(*args,**kwargs)(func)

            if func not in trace_roster:
                trace_roster[name] = {'id': len(trace_roster)}

            func_id = trace_roster[name]['id']
            arg_str = ",".join([arg for arg in arg_set])

            trace_wrapper_source = trace_wrapper_template.format(
                name=name,
                arg_str=arg_str,
                id=func_id,
            )
            exec(trace_wrapper_source,locals(),locals())
            trace_func = eval(f"trace_{func_id}_{name}")
            return numba.njit()(trace_func)
        else:
            return numba.njit(*args,**kwargs)(func)

    return trace_njit_inner



def output_report(mcdc):

    report = open("report.csv","w")
    report.write(f"function name, cpu total runtime (ns), cpu total calls, gpu total runtime (mystery units), gpu total calls\n")
    for name, info in trace_roster.items():
        func_id = info['id']
        cpu_nsecs = mcdc['trace']['slots'][func_id]['runtime_total'][0]
        cpu_calls = mcdc['trace']['slots'][func_id]['call_total'][0]
        gpu_nsecs = mcdc['trace']['slots'][func_id]['runtime_total'][1]
        gpu_calls = mcdc['trace']['slots'][func_id]['call_total'][1]
        report.write(f"{name},{cpu_nsecs},{cpu_calls},{gpu_nsecs},{gpu_calls}\n")
    report.close()

