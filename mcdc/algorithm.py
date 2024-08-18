from numba import njit


@njit
def binary_search(val, grid, length=0):
    """
    Binary search that returns the bin index of the value `val` given grid `grid`.
    Only search up to `length`-th element if `length` is given.

    Some special cases:
        val < min(grid)  --> -1
        val > max(grid)  --> size of bins
        val = a grid point --> bin location whose upper bound is val
                                 (-1 if val = min(grid)
    """
    left = 0
    if length == 0:
        right = len(grid) - 1
    else:
        right = length - 1
    mid = -1
    while left <= right:
        mid = int((left + right) / 2)
        if grid[mid] < val:
            left = mid + 1
        else:
            right = mid - 1
    return int(right)
