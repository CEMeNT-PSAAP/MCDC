import numpy as np

from mcdc.constant import INF
from mcdc.print    import print_error
from mcdc.misc     import binary_search


class MeshCartesian():
    def __init__(self, x=None, y=None, z=None):
        if x is None and y is None and z is None:
            print_error("MeshCartesian at least requires x, y, or z")

        if x is None:
            self.grid_x = np.array([-INF,INF])
        else:
            self.grid_x = x
        
        if y is None:
            self.grid_y = np.array([-INF,INF])
        else:
            self.grid_y = y
        
        if z is None:
            self.grid_z = np.array([-INF,INF])
        else:
            self.grid_z = z

    def get_idx(self, po):
       idx_x = binary_search(po.x, self.grid_x)
       idx_y = binary_search(po.y, self.grid_y)
       idx_z = binary_search(po.z, self.grid_z)
       return idx_x, idx_y, idx_z

    def get_distance(self, po, di):
       idx_x = binary_search(po.x, self.grid_x)
       idx_y = binary_search(po.y, self.grid_y)
       idx_z = binary_search(po.z, self.grid_z)
