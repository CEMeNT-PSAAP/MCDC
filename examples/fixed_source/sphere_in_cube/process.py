import h5py

# Load results
with h5py.File("output.h5", "r") as f:
    # print(f["tallies"].keys())
    print(f["input_deck"]["cell_tallies"].keys())

    for i in range(len(f["input_deck"]["cell_tallies"])):
        fission_score = f[f"tallies/cell_tally_{i}/fission"]

        print(
            f'for sphere {i+1}, mean = {fission_score["mean"][()]}, sdev = {fission_score["sdev"][()]}'
        )

        # print(fission_score["mean"][()])
        # print(fission_score["sdev"][()])

        # print(f"fission_score mean = {fission_score["mean"][()]}")
        # print(f"fission_score mean = {fission_score["sdev"][()]}")

    # cell = f["tallies/cell_tally_0/fission"]
    # print(f'sphere1 mean = {cell["mean"][()]}')
    # print(f'sphere2 sdev = {cell["sdev"][()]}')
