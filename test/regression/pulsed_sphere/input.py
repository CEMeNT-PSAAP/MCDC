import numpy as np
import mcdc

# This regression test uses an approximate pulsed sphere problem to test time dependent cell tallies

exp1 = "C"  # Currently C, Fe supported

# =============================================================================
# Materials
# =============================================================================

mat_air = mcdc.material(
    [
        ["N14", 4.36848e19 * 1e-24],
        ["O16", 1.02529e19 * 1e-24],
    ]
)

if exp1 == "C":
    mat_sphere = mcdc.material(
        [
            ["C12", 9.19e22 * 1e-24],
            ["C13", 9.94e20 * 1e-24],
        ]
    )
    sphere_rad = 20.96
    det_rad = 766.0
elif exp1 == "Fe":
    mat_sphere = mcdc.material(
        [
            ["Fe54", 4.82003e21 * 1e-24],
            ["Fe56", 7.6223e22 * 1e-24],
            ["Fe57", 1.82829e21 * 1e-24],
            ["Fe58", 2.32691e20 * 1e-24],
            ["C12", 1.02809e21 * 1e-24],
            ["Mn55", 8.56743e20 * 1e-24],
            ["P31", 5.9972e20 * 1e-24],
            ["S32", 8.56743e19 * 1e-24],
        ]
    )
    sphere_rad = 4.46
    det_rad = 766.0
else:
    print("Incorrect experiment material provided")

# =============================================================================
# Geometry
# =============================================================================

# Surfaces
s_0 = mcdc.surface("sphere", center=[0.0, 0.0, 0.0], radius=sphere_rad)
s_det_in = mcdc.surface("sphere", center=[0.0, 0.0, 0.0], radius=det_rad - 0.5)
s_det_out = mcdc.surface("sphere", center=[0.0, 0.0, 0.0], radius=det_rad + 0.5)
s_ext = mcdc.surface("sphere", center=[0.0, 0.0, 0.0], radius=1000, bc="vacuum")

# Cells
c_iron_sphere = mcdc.cell(-s_0, mat_sphere)
c_air1 = mcdc.cell(+s_0 & -s_det_in, mat_air)
c_det1 = mcdc.cell(+s_det_in & -s_det_out, mat_air)
c_air2 = mcdc.cell(+s_det_out & -s_ext, mat_air)


# =============================================================================
# Source
# =============================================================================

mcdc.source(
    point=[0.0, 0.0, 0.0],
    energy=np.array(
        [
            [
                14.00e6,
                14.10e6,
                14.20e6,
                14.30e6,
                14.40e6,
                14.50e6,
                14.60e6,
                14.70e6,
                14.80e6,
                14.90e6,
                15.00e6,
                15.10e6,
                15.20e6,
                15.30e6,
                15.40e6,
                15.50e6,
            ],
            [
                0.0000,
                0.0000,
                0.3960,
                3.9776,
                18.2301,
                40.7022,
                40.1637,
                31.6909,
                23.2557,
                17.6923,
                15.2267,
                10.3729,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
        ]
    ),
    isotropic=True,
    time=np.array([-1.5 * 1e-9, 1.5e-9]),
)

#    energy=np.array([
#        [12.80e+6, 12.90e+6, 13.00e+6, 13.10e+6, 13.20e+6, 13.30e+6, 13.40e+6, 13.50e+6, 13.60e+6, 13.70e+6, 13.80e+6, 13.90e+6,],
#        [0.0000, 0.0000, 0.0000, 6.3499, 22.6144, 31.9989, 46.6420, 49.5506, 25.0656, 4.0393, 0.2414, 0.0000]
#    ]),

# Shift source?
#    energy = np.array([
#        [14.00e6, 14.10e6, 14.20e6, 14.30e6, 14.40e6, 14.50e6, 14.60e6, 14.70e6, 14.80e6, 14.90e6, 15.00e6, 15.10e6, 15.20e6, 15.30e6, 15.40e6, 15.50e6],
#        []0.0000, 0.0000,   0.3960,   3.9776,     18.2301,  40.7022,  40.1637,  31.6909,       23.2557,  17.6923,  15.2267,  10.3729,        0.0000,   0.0000,   0.0000,   0.0000]
#    ]),

# =============================================================================
# Tallies
# =============================================================================
PStally = mcdc.tally.mesh_tally(
    scores=["flux"],
    x=[740.0, 760.0],
    y=[-10.0, 10.0],
    z=[-10.0, 10.0],
    t=np.linspace(0.0, 500.0e-9, 250),
)

full_sphere_tally = mcdc.tally.cell_tally(
    c_det1,
    t=np.linspace(0.0, 500.0e-9, 250),
    # E=np.linspace(0.0,20e6,20),
    scores=["flux"],
)

# =============================================================================
# Settings
# =============================================================================

mcdc.setting(N_particle=1e3)
mcdc.time_census(np.linspace(0.0, 500.0e-9, 125)[1:-1])
mcdc.implicit_capture()

mcdc.run()
