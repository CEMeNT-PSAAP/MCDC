def binary_search(val, grid):
    """
    Binary search that returns the bin index of a value `val` given
    grid `grid`
    
    Some special cases:
        `val` < min(`grid`)  --> -1
        `val` > max(`grid`)  --> size of bins
        `val` = a grid point --> bin location whose upper bound is `val`
                                 (-1 if val = min(grid)
    """
    
    left  = 0
    right = len(grid) - 1
    mid   = -1
    while left <= right:
        mid = (int((left + right)/2))
        if grid[mid] < val: left = mid + 1
        else:            right = mid - 1
    return int(right)


def interpolate(x, x1, x2, y1, y2):
    return (x-x2)/(x1-x2)*y1 + (x-x1)/(x2-x1)*y2
