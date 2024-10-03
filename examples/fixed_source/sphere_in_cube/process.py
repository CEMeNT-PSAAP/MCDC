import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.colors as colors

# Load results
with h5py.File("output.h5", "r") as f:
    print("Keys: %s" % f.keys())
    tallies = f["tallies"]
    print("Tallies keys: %s" %tallies.keys())
    print("Cell keys: %s" %tallies["cell_tally_0"].keys())
    print("Fission keys: %s" %tallies["cell_tally_0"]["fission"].keys())

    cell = f["tallies/cell_tally_0/fission/mean"][()]
    print(f'cell tally = {cell}')