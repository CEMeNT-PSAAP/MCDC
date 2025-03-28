import numba
from numba.types import uint64
import numpy as np
from mcdc.constant import *

# =============================================================================
# Random number generator
#   LCG with hash seed-split
# =============================================================================


@numba.njit()
def wrapping_mul(a, b):
    return a * b


@numba.njit()
def wrapping_add(a, b):
    return a + b


def wrapping_mul_python(a, b):
    a = uint64(a)
    b = uint64(b)
    with np.errstate(all="ignore"):
        return a * b


def wrapping_add_python(a, b):
    a = uint64(a)
    b = uint64(b)
    with np.errstate(all="ignore"):
        return a + b


def adapt_rng(object_mode=False):
    global wrapping_add, wrapping_mul
    if object_mode:
        wrapping_add = wrapping_add_python
        wrapping_mul = wrapping_mul_python


@numba.njit()
def split_seed(key, seed):
    """murmur_hash64a"""
    multiplier = uint64(0xC6A4A7935BD1E995)
    length = uint64(8)
    rotator = uint64(47)
    key = uint64(key)
    seed = uint64(seed)

    hash_value = uint64(seed) ^ wrapping_mul(length, multiplier)

    key = wrapping_mul(key, multiplier)
    key ^= key >> rotator
    key = wrapping_mul(key, multiplier)
    hash_value ^= key
    hash_value = wrapping_mul(hash_value, multiplier)

    hash_value ^= hash_value >> rotator
    hash_value = wrapping_mul(hash_value, multiplier)
    hash_value ^= hash_value >> rotator
    return hash_value


@numba.njit()
def rng_(seed):
    seed = uint64(seed)
    return wrapping_add(wrapping_mul(RNG_G, seed), RNG_C) & RNG_MOD_MASK


@numba.njit()
def rng(state_arr):
    state = state_arr[0]
    state["rng_seed"] = rng_(state["rng_seed"])
    return state["rng_seed"] / RNG_MOD


@numba.njit()
def rng_from_seed(seed):
    return rng_(seed) / RNG_MOD


@numba.njit()
def rng_array(seed, shape, size):
    xi = np.zeros(size)
    for i in range(size):
        xi_seed = split_seed(i, seed)
        xi[i] = rng_from_seed(xi_seed)
    xi = xi.reshape(shape)
    return xi


def bits_to_uint(arg):
    kind = numba.typeof(arg)
    if kind == numba.uint8:
        return arg
    elif kind == numba.uint16:
        return arg
    elif kind == numba.uint32:
        return arg
    elif kind == numba.uint64:
        return arg
    elif kind == numba.int8:
        conv = np.int8(arg)
        return conv.view(np.uint8)
    elif kind == numba.int16:
        conv = np.int16(arg)
        return conv.view(np.uint16)
    elif kind == numba.int32:
        conv = np.int32(arg)
        return conv.view(np.uint32)
    elif kind == numba.int64:
        conv = np.int64(arg)
        return conv.view(np.uint64)
    elif kind == numba.float32:
        conv = np.float32(arg)
        return conv.view(np.uint32)
    elif kind == numba.float64:
        conv = np.float64(arg)
        return conv.view(np.uint64)
    elif kind == numba.bool_:
        if arg:
            return numba.uint8(1)
        else:
            return numba.uint8(0)
    else:
        return numba.uint8(0)



@numba.core.extending.overload(bits_to_uint)
def overload_bits_to_uint(kind):
    if kind == numba.uint8:
        def inner(arg):
            return arg
    elif kind == numba.uint16:
        def inner(arg):
            return arg
        return inner
    elif kind == numba.uint32:
        def inner(arg):
            return arg
        return inner
    elif kind == numba.uint64:
        def inner(arg):
            return arg
        return inner
    elif kind == numba.int8:
        def inner(arg):
            return arg.view(numba.uint8)
        return inner
    elif kind == numba.int16:
        def inner(arg):
            return arg.view(numba.uint16)
        return inner
    elif kind == numba.int32:
        def inner(arg):
            return arg.view(numba.uint32)
        return inner
    elif kind == numba.int64:
        def inner(arg):
            return arg.view(numba.uint64)
        return inner
    elif kind == numba.float32:
        def inner(arg):
            return arg.view(numba.uint32)
        return inner
    elif kind == numba.float64:
        def inner(arg):
            return arg.view(numba.uint64)
        return inner
    elif kind == numba.bool_:
        def inner(arg):
            if arg:
                return numba.uint8(1)
            else:
                return numba.uint8(0)
        return inner
    else:
        return numba.uint8(0)

def hash_data(arg):
    kind = numba.typeof(arg)
    if isinstance(kind,numba.types.Record):
        kind_dtype = numba.np.numpy_support.as_dtype(kind)
        keys = tuple(kind_dtype.fields.keys())
        result = numba.uint64(0)
        if kind_dtype.itemsize > 1024:
            return result
        mask   = numba.uint64(0x7FFFFFFFFFFFFFFF)
        for key in numba.literal_unroll(keys):
            result = split_seed(result,hash_data(arg[key]))
        return result
    elif isinstance(kind,numba.types.Array):
        result = numba.uint64(0)
        mask   = numba.uint64(0x7FFFFFFFFFFFFFFF)
        kind_dtype = numba.np.numpy_support.as_dtype(kind.dtype)
        if kind_dtype.itemsize*len(arg) > 1024:
            return result
        
        for value in arg:
            result = split_seed(result,hash_data(value) & mask)
        return numba.uint64(result)
    elif isinstance(kind,numba.types.Integer):
        result =  rng_(bits_to_uint(arg))
        return result
    elif isinstance(kind,numba.types.Float):
        result =  rng_(bits_to_uint(arg))
        return result
    elif isinstance(kind,numba.types.Boolean):
        result =  rng_(bits_to_uint(arg))
        return result
    else:
        return numba.uint64(0)

@numba.core.extending.overload(hash_data)
def overload_hash_data(arg):
    print(f"KIND: {arg}")
    kind = arg
    if isinstance(kind,numba.types.Record):
        kind_dtype = numba.np.numpy_support.as_dtype(kind)
        keys = tuple(kind_dtype.fields.keys())
        if kind_dtype.itemsize > 1024:
            def inner(arg):
                return numba.uint64(0)
            return inner
        else:
            def inner(arg):
                result = numba.uint64(0)
                mask   = numba.uint64(0x7FFFFFFFFFFFFFFF)
                for key in numba.literal_unroll(keys):
                    result = split_seed(result,hash_data(arg[key]))
                return result
            return inner
    elif isinstance(kind,numba.types.Array):
        kind_dtype = numba.np.numpy_support.as_dtype(kind.dtype)
        if kind_dtype.itemsize*len(arg) > 1024:
            def inner(arg):
                return numba.uint64(0)
            return inner
        else:
            def inner(arg):
                result = numba.uint64(0)
                mask   = numba.uint64(0x7FFFFFFFFFFFFFFF)
                for value in arg:
                    result = split_seed(result,hash_data(value) & mask)
                return numba.uint64(result)
            return inner
    elif isinstance(kind,numba.types.Integer):
        def inner(arg):
            result =  rng_(bits_to_uint(arg))
            return result
        return inner
    elif isinstance(kind,numba.types.Float):
        def inner(arg):
            result =  rng_(bits_to_uint(arg))
            return result
        return inner
    elif isinstance(kind,numba.types.Boolean):
        def inner(arg):
            result =  rng_(bits_to_uint(arg))
            return result
        return inner
    else:
        def inner(arg):
            return numba.uint64(0)
        return inner



