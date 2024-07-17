# Analytic Tests
To run these tests, navigate to this folder in the command line, then run `run.py` with whichever decorators are appropriate to your machine and intended method of running.
Then, run process.py, which creates convergence plots saved in the results folder. Finally, run display.py to show these plots. To change the limits on and number of tests when it
comes to the number of simulated particles, use `task.py` - the program will run with particle amounts between the limits at `"N"` equally spaced (in log space) points.

## AZURV1 Variants
These tests (azurv1_###) are implementations of MC/DC compared to benchmarks computed semi-analytically according to this [paper](https://www.tandfonline.com/doi/abs/10.1080/23324309.2022.2103151) (all subsequent equation numbers reference this).

Each tests convergence for flux comprehensively measured across the specified x and t ranges (-20.5 to 20.5 and 0 to 20 respectively). `plane_IC` tests an isotropic plane pulse at x=t=0,
`square_IC` tests a square pulse from x = -.5 to x = .5 at t = 0. `square_source` tests the same but the source is on from t = 0 to t = 5. `gaussian_IC` and `gaussian_source` follow the same
principle, except with a manufactured gaussian distribution instead of the square.

Reference values are precomputed, as the benchmark file becomes prohibitively large when containing sufficient solutions (the reference values are integrations of many instantaneous 
solutions in a benchmarks file)

To compute these benchmarks and use them with the `makereference.py` files supplied, download and use the code found [here](https://github.com/wbennett39/transport_benchmarks.git) as follows:

Navigate to the `transport_benchmarks` folder and run `python -i run.py`
    
`x = np.linspace(-20.5,20.5,202)`

`dx = x[1:]-x[:-1]`

`x = x[1:]-dx/2`

#### For Planar Pulse:

`t = np.linspace(0,20,100001)`

`for t in t:`

`    greens.plane_IC(t = t, npnts = 201, c = 1.0, choose_xs = True, xpnts = x)`

This uses equations [5] and [7] in the paper

#### For Square Pulse:

`t = np.linspace(0,20,100001)`

`for t in t:`

`    greens.square_IC(t = t, npnts = 201, x0 = 0.5, c = 1.0, choose_xs = True, xpnts = x)`

This uses equations [14] and [15] in the paper

#### For Square Source:

`t = np.linspace(0,20,10001)` Note this difference from previous solutions

`for t in t:`

`    greens.square_source(t = t, x0 = 0.5, t0 = 5, npnts = 201, c = 1.0, choose_xs = True, xpnts = x)`

This uses equations [26] and [32] in the paper

#### For Gaussian Pulse:

`t = np.linspace(0,20,10001)` 

`for t in t:`

`    greens.gaussian_IC(t = t, sigma = 0.5, npnts = 201, c = 1.0, choose_xs = True, xpnts = x)`

This uses equations [19] and [21] in the paper

#### For Gaussian Source:

`t = np.linspace(0,20,10001)`

`for t in t:`

`    greens.gaussian_source(t = t, sigma = 0.5, t0 = 5, npnts = 201, c = 1.0, choose_xs = True, xpnts = x)`

This uses equations [35] and [37] in the paper
