import numpy as np
import h5py
import mcdc
import pathlib
import matplotlib.pyplot as plt
# Tests convergence to benchmark from semi-analytic solutions of William Bennett for various source types

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic source at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1



def runtest(particlenums,sourcetype,loud = True,createdata = True):
    '''
    runs tests at time 1, 5, and 10
    particlenums : points at which to test, where the point is a number of particles
    sourcetype : type of source ("square_IC","square_source","gaussian_IC","gaussian_source", or "all")
    '''
    all = False
    if sourcetype == "all":
        all = True
        sourcetype = "plane_IC"
    
    if createdata:
        for i in particlenums:
            # Set materials
            m = mcdc.material(
                capture=np.array([1.0 / 3.0]),
                scatter=np.array([[1.0 / 3.0]]),
                fission=np.array([1.0 / 3.0]),
                nu_p=np.array([2]),
            )
            if sourcetype == "plane_IC":
                mcdc.source(point=[0.0,0.0,0.0], isotropic=True, time=[1e-10, 1e-10])
            elif sourcetype == "square_IC":
                mcdc.source(x=[-.5,.5], isotropic=True, time=[1e-10, 1e-10])
            elif sourcetype == "square_source":
                mcdc.source(x=[-.5,.5], isotropic=True, time=[1e-10, 5])
            elif sourcetype == "gaussian_IC":
                x = np.linspace(-15,15,int(1e4))
                gaussian = np.exp(-4*x**2)
                dx = x[2]-x[1]
                edges_x = np.append(x-dx/2,x[-1]+dx/2)
                for ii, (x1,x2) in enumerate(zip(edges_x[:-1],edges_x[1:])):
                    mcdc.source(x=[x1,x2],prob=gaussian[ii],time=[1e-10, 1e-10])
            elif sourcetype == "gaussian_source":
                x = np.linspace(-15,15,int(1e4))
                gaussian = np.exp(-4*x**2)
                dx = x[2]-x[1]
                edges_x = np.append(x-dx/2,x[-1]+dx/2)
                for ii, (x1,x2) in enumerate(zip(edges_x[:-1],edges_x[1:])):
                    mcdc.source(x=[x1,x2],prob=gaussian[ii],time=[1e-10, 5])

            # Set surfaces
            s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
            s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

            # Set cells
            mcdc.cell([+s1, -s2], m)

            # =============================================================================
            # Set source
            # =============================================================================
            # Isotropic pulse at x=t=0
            
           
            # =============================================================================
            # Set tally, setting, and run mcdc
            # =============================================================================

            # Tally: cell-average, cell-edge, and time-edge scalar fluxes
            x = np.linspace(-10,10,1001)
            dx = x[1] - x[0]
            x = np.append(x-dx/2, x[-1]+dx/2)
            mcdc.tally(
                scores=["flux"],
                x = x,
                t=np.linspace(0, 10, 10001),
                )

            # Setting

            mcdc.setting(N_particle = i)
            mcdc.setting(output_name=sourcetype+"p"+str(i)+"t1e-3")

            # Run
            mcdc.run()
    if all:
        for s in ["square_IC","square_source","gaussian_IC","gaussian_source"]:
            runtest(particlenums,s,loud,createdata)
    
    phi = [None]*len(particlenums)
    phi_sd = [None]*len(particlenums)
    for i in range(len(particlenums)):
        with h5py.File(str(pathlib.Path().resolve())+"/"+sourcetype+"p"+str(particlenums[i])+"t1e-3.h5", "r") as f:
            x = f["tally/grid/x"][:]
            dx = x[1:] - x[:-1]
            x_mid = 0.5 * (x[:-1] + x[1:])
            t = f["tally/grid/t"][:]
            dt = t[1:] - t[:-1]
            K = len(t) - 1

            phi[i] = f["tally/flux/mean"][:]
            phi_sd[i] = f["tally/flux/sdev"][:]
            

        # Normalize
        for k in range(K):
            phi[i][k] /= dx * dt[k]
            phi_sd[i][k] /= dx * dt[k]
            if sourcetype == "gaussian_source":
                phi[i][k] *= 5*0.8862269254527580136490837416705725913987747280611935641069038949
                phi_sd[i][k] *= 5*0.8862269254527580136490837416705725913987747280611935641069038949
            if sourcetype == "gaussian_IC":
                phi[i][k] *= 0.8862269254527580136490837416705725913987747280611935641069038949
                phi_sd[i][k] *= 0.8862269254527580136490837416705725913987747280611935641069038949
            if sourcetype == "square_source":
                phi[i][k] *= 5
                phi_sd[i][k] *= 5
    errors = [1000]*len(particlenums)
    error = lambda phi, y: np.sqrt(np.mean((phi-y)**2))
    with h5py.File(str(pathlib.Path(__file__).parent.resolve())+"/benchmarks.hdf5","r") as f:
        extra = ""
        if 'plane' not in sourcetype:
            extra="x0=0.5" 
        data1 = np.array(f[sourcetype+"/t = 1"+extra][:])
        data5 = np.array(f[sourcetype+"/t = 5"+extra][:])
        data10 = np.array(f[sourcetype+"/t = 10"+extra][:])
        y1 = data1[1,:]
        y5 = data5[1,:]
        y10 = data10[1,:]
    for i in range(3):
        for j in range(len(particlenums)):
            y = [y1,y5,y10][i]
            index = [1000,5000,9999][i]
            errors[j] = error(phi[j][index],y)
        if loud:
            
            plt.figure()
            plt.loglog(particlenums,errors,'-o',label="Error timestep 1e-3")
            plt.loglog(particlenums,1/np.sqrt(particlenums), label=r'O(1/$\sqrt{N}$)')
            plt.title("Convergence for "+sourcetype+" at t = "+["1","5","10"][i])
            plt.xlabel("Number of particles")
            plt.ylabel("Error")
            plt.legend()
            #plt.figure() this and next two lines are sanity checks to see that plots are lining up
            #plt.plot(x_mid,phi[2][5000])
            #plt.plot(x_mid,y5)
    m,b = np.polyfit(np.log(particlenums),np.log(errors),1)
    return m


def test(particlenums,sourcetype,loud = True,createdata = True,tol = .1):
    '''
    shows plots (if loud==true) of convergence from previously defined runtest, asserts that convergence is O(1/sqrt(N)) to tolerance tol
    '''

    slope = runtest(particlenums,sourcetype,loud,createdata)
    assert np.fabs(slope+1/2) < tol

    plt.show(block = False)
    plt.pause(.001)
    input("hit[enter] to close all plots.")
    plt.close('all') # all open plots are correctly closed after each run

### here is where your input goes (unless you call these functions from somewhere else)

testparticles = [1e2,1e3,1e4]
sourcetype = "all"
loud = False
createdata = True
tol = .1

test(testparticles,sourcetype,loud,createdata,tol)

# for speed, run in CL with --mode=numba as per usual
# do not run mpirun with loud on - the plots will throw an error    


