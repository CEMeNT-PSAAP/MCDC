def binary_search(val, grids):
    # Return bin index
    #   val < min(grids) --> -1
    #   val > max(grids) --> # of bins
    #   val = gird point --> bin location whose upper bound is val
    #                        (-1 if val = min(grids)
    
    left  = 0
    right = len(grids) - 1
    mid   = -1
    while left <= right:
        mid = (int((left + right)/2))
        if grids[mid] < val: left = mid + 1
        else:            right = mid - 1
    return int(right)


def interpolate(x, x1, x2, y1, y2):
    return (x-x2)/(x1-x2)*y1 + (x-x1)/(x2-x1)*y2