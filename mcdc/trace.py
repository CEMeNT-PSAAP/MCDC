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
int64_t mono_clock () {
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


def calculate_hash(arg):
    if isinstance(arg,numba.types.Hashable):
        return hash(arg)
    else:
        return 0

@numba.core.extending.overload(calculate_hash)
def caluclate_hash_overload(arg):
    if not isinstance(arg,numba.types.Hashable):
        def calc(arg):
            return 0
        return calc
    else:
        def calc(arg):
            return numba.uint64(hash(arg))
        return calc


def generate_trace_hash_fn(id,name,arg_list,arg_str,trace_state_extractor):
    code = "@numba.njit()\n"
    code += f"def trace_hash_{id}_{name} ({arg_str}):\n"
    code += f"    {trace_state_extractor}\n"
    code += f"    result = numba.uint64(0)\n"
    for arg in arg_list:
        code += f"    result = result ^ numba.uint64(calculate_hash({arg}))\n"
    code += f"    return result\n"
    return code




trace_roster = {}

trace_wrapper_template = """
def trace_{id}_{name} ({arg_str}) :
    {trace_state_extractor}
    hash = trace_hash_{id}_{name}({arg_str})
    log_hash(trace,hash)
    old_id = get_func_id(trace)
    old_depth = get_depth(trace)
    set_func_id(trace,{id})
    set_depth(trace,old_depth+1)
    t0 = trace_get_clock()
    result = func ({arg_str})
    t1 = trace_get_clock()
    set_func_id(trace,old_id)
    set_depth(trace,old_depth)
    platform_index = trace_platform_index()
    adapt.global_add(trace['slots'][{id}]['runtime_total'],platform_index, t1 - t0)
    adapt.global_add(trace['slots'][{id}]['call_total'],platform_index,1)
    return result
"""

trace_wrapper_name_template = "trace_{name}"



sig     = numba.core.typing.signature
ext_fn  = numba.types.ExternalFunction
gpu_get_wall_clock = ext_fn("get_wall_clock",sig(numba.types.int64))

sig     = numba.core.typing.signature
ext_fn  = numba.types.ExternalFunction
get_integer_thread_id = ext_fn("get_integer_thread_id",sig(numba.types.int64))


###############################################################################
# Wall clock for measuring runtime
###############################################################################


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



###############################################################################
# Index associated with each target
###############################################################################


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



###############################################################################
# Thread id for indexing per-thread tracing info
###############################################################################

def thread_id():
    return 0

@numba.core.extending.overload(thread_id, target="cpu")
def cpu_thread_id():
    def inner_thread_id():
        return 0
    return inner_thread_id

@numba.core.extending.overload(thread_id, target="gpu")
def gpu_thread_id():
    def inner_thread_id():
        return 0
    return inner_thread_id





###############################################################################
# Stack id functions
###############################################################################

@numba.njit
def alloc_stack_id(trace):
    stack_id = adapt.global_add(trace['thread_state']['stack_id_offset'],0,1)
    trace['thread_state'][thread_id()]['stack_id'] = stack_id

@numba.njit
def get_stack_id(trace):
    return trace['thread_state'][thread_id()]['stack_id']





###############################################################################
# Function id functions
###############################################################################

@numba.njit
def set_func_id(trace,func_id):
    trace['thread_state'][thread_id()]['func_id'] = func_id

@numba.njit
def get_func_id(trace):
    return trace['thread_state'][thread_id()]['func_id']

###############################################################################
# Depth functions
###############################################################################

@numba.njit
def set_depth(trace,depth):
    trace['thread_state'][thread_id()]['depth'] = depth

@numba.njit
def get_depth(trace):
    return trace['thread_state'][thread_id()]['depth']

###############################################################################
# Previous fingerprint slot
###############################################################################

@numba.njit
def set_prev_fingerprint_slot(trace,prev_slot):
    trace['thread_state'][thread_id()]['previous_slot'] = prev_slot

@numba.njit
def get_prev_fingerprint_slot(trace):
    return trace['thread_state'][thread_id()]['previous_slot']

###############################################################################
# Records hash and associated metadata to tracing data structures
###############################################################################

@numba.njit
def log_hash(trace,hash_value):
    offset = adapt.global_add(trace['fingerprint_offset'],0,1)
    if offset >= trace['fingerprint_slot_limit']:
        return
    trace['fingerprints'][offset]['stack_id'] = get_stack_id(trace)
    trace['fingerprints'][offset]['func_id']  = get_func_id(trace)
    trace['fingerprints'][offset]['depth']    = get_depth(trace)
    trace['fingerprints'][offset]['hash']     = 0





###############################################################################
# Stack id functions
###############################################################################

def trace(transforms=[]):

    def trace_inner(func):
        global get_clock
        global platform_index
        trace_get_clock = get_clock
        trace_platform_index = platform_index
        log_hash     = globals()['log_hash']
        set_func_id  = globals()['set_func_id']
        get_func_id  = globals()['get_func_id']
        get_stack_id = globals()['get_stack_id']
        get_depth    = globals()['get_depth']
        set_depth    = globals()['set_depth']
        calculate_hash = globals()['calculate_hash']

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
            import numba

            for tr in transforms:
                func = tr(func)

            if func not in trace_roster:
                trace_roster[name] = {'id': len(trace_roster)}

            func_id = trace_roster[name]['id']
            arg_list = [arg for arg in arg_set]
            arg_str = ",".join(arg_list)

            trace_wrapper_source = trace_wrapper_template.format(
                name=name,
                arg_str=arg_str,
                id=func_id,
                trace_state_extractor=trace_state_extractors[extractor_target]
            )
            hash_fn_code = generate_trace_hash_fn(
                func_id,
                name,
                arg_list,
                arg_str,
                trace_state_extractors[extractor_target]
            )
            exec(hash_fn_code,locals(),locals())
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


def dd_mergetrace(mcdc):
    d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1
    i = 0
    for n in range(d_Nx * d_Ny * d_Nz):
        dd_ranks = []
        for r in range(int(mcdc["technique"]["dd_work_ratio"][n])):
            dd_ranks.append(i)
            i += 1
        # create MPI Comm group out of subdomain processors
        dd_group = MPI.COMM_WORLD.group.Incl(dd_ranks)
        dd_comm = MPI.COMM_WORLD.Create(dd_group)
        # MPI Reduce on subdomain processors
        for name, info in trace_roster.items():
            func_id = info['id']

            if MPI.COMM_NULL != dd_comm:
                python_nsecs = dd_comm.reduce(mcdc['trace']['slots'][func_id]['runtime_total'][0], MPI.SUM)
                python_calls = dd_comm.reduce(mcdc['trace']['slots'][func_id]['call_total'][0], MPI.SUM)
                cpu_nsecs = dd_comm.reduce(mcdc['trace']['slots'][func_id]['runtime_total'][1], MPI.SUM)
                cpu_calls = dd_comm.reduce(mcdc['trace']['slots'][func_id]['call_total'][1], MPI.SUM)
                if mcdc["dd_local_rank"] == 0:
                    mcdc['trace']['slots'][func_id]['runtime_total'][0] = python_nsecs
                    mcdc['trace']['slots'][func_id]['call_total'][0] = python_calls
                    mcdc['trace']['slots'][func_id]['runtime_total'][1] = cpu_nsecs
                    mcdc['trace']['slots'][func_id]['call_total'][1] = cpu_calls

        # free comm group
        dd_group.Free()
        if MPI.COMM_NULL != dd_comm:
            dd_comm.Free()

def initialize(mcdc):
    mcdc['trace']['slot_limit'] = config.trace_slot_limit
    mcdc['trace']['fingerprint_slot_limit'] = config.trace_slot_limit


def output_report(mcdc):

    if not mcdc["technique"]["domain_decomposition"]:
        report = open("report.csv","w")
        report.write("function name, ")
        report.write("python total runtime (ns), python total calls, ")
        report.write("cpu total runtime (ns), cpu total calls, ")
        report.write("gpu total runtime (mystery units), gpu total calls, ")
        report.write("\n")

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

    else: # write report for each subdomain
        dd_mergetrace(mcdc)
        d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
        d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
        d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

        i = 0
        for n in range(d_Nx * d_Ny * d_Nz):
            if mcdc["dd_local_rank"] == 0 and mcdc["dd_idx"] == n:
                report_name = f"report{n}.csv"
                report = open(report_name, "w")
                report.write("function name, ")
                report.write("python total runtime (ns), python total calls, ")
                report.write("cpu total runtime (ns), cpu total calls, ")
                report.write("gpu total runtime (mystery units), gpu total calls, ")
                report.write("\n")

                gpu_rate = 1000000000
                if config.target == "gpu":
                    gpu_rate = gpu_clock_rate()

                for name, info in trace_roster.items():
                    func_id = info['id']
                    slot = mcdc['trace']['slots'][func_id]

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


def output_fingerprints(mcdc):

    if not mcdc["technique"]["domain_decomposition"]:
        prints = open("fingerprints.csv","w")
        for idx in range(mcdc['trace']['fingerprint_offset'][0]):
            fp = mcdc['trace']['fingerprints'][idx]
            prints.write(f"{fp}\n")
        prints.close()
    else:
        my_rank = MPI.Get_rank()
        for rank in range(MPI.Get_size()):
            if rank != my_rank:
                continue
            if rank == 0:
                prints = open("fingerprints.csv","w")
            else:
                prints = open("fingerprints.csv","a")


            prints.close()


