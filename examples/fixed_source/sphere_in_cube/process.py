import h5py

# Load results
with h5py.File("output.h5", "r") as f:
    cell = f["tallies/cell_tally_0/fission"]
    print(f'sphere mean = {cell["mean"][()]}')    
    print(f'sphere sdev = {cell["sdev"][()]}')