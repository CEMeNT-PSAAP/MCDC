
import numba



@numba.extending.overload(np.dot,target="gpu")
def my_gpu_dot(a,b):

    if ! isinstance(a,numba.types.Array):
        return None

    if ! isinstance(b,numba.types.Array):
        return None

    def impl(a,b):
        n = min(len(a),len(b))
        result = 0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result

    return impl

