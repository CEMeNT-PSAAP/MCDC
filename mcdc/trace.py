import inspect
import mcdc.config as config
import numba
import ctypes
import time





trace_roster = {}

trace_wrapper_template = """
def trace_{id}_{name} ({arg_str}) :
    t0 = inner_get_clock()
    result = njit_func ({arg_str})
    t1 = inner_get_clock()
    mcdc['trace']['slots'][{id}]['runtime'] = t1 - t0
    return result
"""

trace_wrapper_name_template = "trace_{name}"


get_system_clock = ctypes.pythonapi._PyTime_GetSystemClock
get_system_clock.argtypes = []
get_system_clock.restype = ctypes.c_int64

@numba.njit()
def get_clock():
    print(get_system_clock())
    return 1



def njit(*args,**kwargs):

    def trace_njit_inner(func):
        inner_get_clock = get_clock

        result_func = func

        if config.trace :

            global trace_roster

            njit_func = numba.njit(*args,**kwargs)(func)

            if func not in trace_roster:
                trace_roster[func] = {'id': len(trace_roster)}

            func_id = trace_roster[func]['id']

            name = func.__name__
            arg_set = inspect.signature(func).parameters
            arg_str = ",".join([arg for arg in arg_set])
            trace_wrapper_source = trace_wrapper_template.format(
                name=name,
                arg_str=arg_str,
                id = func_id
            )

            exec(trace_wrapper_source,locals(),locals())
            print(trace_wrapper_source)

            result_func = eval(f"trace_{func_id}_{name}")

        return numba.njit(*args,**kwargs)(result_func)

    return trace_njit_inner



def print_report(mcdc):

    for func, info in trace_roster.items():
        name = func.__name__
        func_id = info['id']
        nsecs = mcdc['trace']['slots'][func_id]['runtime']
        print(f"{name} : {nsecs}ns")


