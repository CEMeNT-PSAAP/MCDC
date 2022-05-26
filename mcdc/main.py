import h5py
import numba as nb
import numpy as np

import mcdc.mpi   as mpi
import mcdc.type_ as type_

from mcdc.class_.particle import type_particle
from mcdc.class_.popctrl  import PCT_CO
from mcdc.constant        import *
from mcdc.looper          import loop_source
from mcdc.print_          import print_banner, print_msg, print_runtime,\
                                 print_progress_eigenvalue
from mcdc.util            import profile

# Get mcdc global variables as "mcdc"
import mcdc.global_ as mcdc_
mcdc = mcdc_.global_

#@profile
def run():
    # Print banner and configuration
    print_banner()
    if nb.config.DISABLE_JIT:
        print_msg("           Mode | Python")
    else:
        print_msg("           Mode | Numba")
    print_msg("  MPI Processes | %i"%mpi.size)
    print_msg(" OpenMP Threads | 1") # TODO

    # Preparation (memory allocation, etc)
    prepare()
    
    # Run
    print_msg(" Now running TNT...")
    
    # Start timer
    mcdc.runtime_total = mpi.Wtime()

    # SIMULATION LOOP
    simulation_end = False
    while not simulation_end:
        # Loop over source particles
        loop_source(mcdc)
        
        # Eigenvalue mode generation closeout
        if mcdc.setting.mode_eigenvalue:
            tally_closeout()
            tally_global_closeout()
            print_progress_eigenvalue(mcdc)

        # Simulation end?
        if mcdc.setting.mode_eigenvalue:
            mcdc.i_iter += 1
            if mcdc.i_iter == mcdc.setting.N_iter: simulation_end = True
        elif mcdc.bank_census['size'] == 0: 
            simulation_end = True

        # Manage particle banks
        if not simulation_end:
            t = mpi.Wtime()
            manage_particle_banks()
            t = mpi.Wtime() - t
            mcdc.runtime_pct += t
            
    # Fixed-source mode closeout
    if not mcdc.setting.mode_eigenvalue:
        tally_closeout()

    # Stop timer
    mcdc.runtime_total = mpi.Wtime() - mcdc.runtime_total
    
    # Generate output file
    generate_hdf5()
    
    print_runtime(mcdc)

    mcdc.reset()

def prepare():
    print_msg("\n Preparing...")

    # Tally
    mcdc.tally.allocate_bins(mcdc.cells[0].material.G, mcdc.setting.N_iter)

    # Create particle banks
    Nmax = mcdc.setting.Nmax
    Nmax_census = mcdc.setting.Nmax_census
    Nmax_source = mcdc.setting.Nmax_source
    mcdc.bank_history = type_.make_bank('history', Nmax)
    if mcdc.setting.mode_eigenvalue:
        mcdc.bank_census  = type_.make_bank('census', 
                                            mcdc.setting.N_hist*Nmax_census)
        mcdc.bank_source  = type_.make_bank('source', 
                                            mcdc.setting.N_hist*Nmax_source)
        mcdc.bank_fission = mcdc.bank_census
    else:
        mcdc.bank_census  = type_.make_bank('census', 0)
        mcdc.bank_source  = type_.make_bank('source', 0)
        mcdc.bank_fission = mcdc.bank_history

    # Population control
    if mcdc_.population_control is None:
        mcdc_.population_control = PCT_CO()
    mcdc_.population_control.prepare(mcdc.setting.N_hist)

    # Normalize source probabilities
    tot = 0.0
    for S in mcdc.sources:
        tot += S.prob
    for S in mcdc.sources:
        S.prob /= tot

    # Distribute work to processors
    mpi.distribute_work(mcdc.setting.N_hist)
    mcdc.N_work = mpi.work_size
    mcdc.master = mpi.master


def tally_closeout():
    if mcdc.tally.flux:
        score_closeout(mcdc.tally.score_flux)
    if mcdc.tally.current:
        score_closeout(mcdc.tally.score_current)
    if mcdc.tally.eddington:
        score_closeout(mcdc.tally.score_eddington)
    if mcdc.tally.flux_x:
        score_closeout(mcdc.tally.score_flux_x)
    if mcdc.tally.flux_t:
        score_closeout(mcdc.tally.score_flux_t)
    if mcdc.tally.current_x:
        score_closeout(mcdc.tally.score_current_x)

def score_closeout(score):
    N_hist = mcdc.setting.N_hist
    i_iter = mcdc.i_iter

    # MPI Reduce
    score.sum_[:]   = mpi.reduce_master(score.sum_)
    score.sum_sq[:] = mpi.reduce_master(score.sum_sq)
    
    # Store results
    score.mean[i_iter,:] = score.sum_/N_hist
    score.sdev[i_iter,:] = np.sqrt((score.sum_sq/N_hist 
                                - np.square(score.mean[i_iter]))\
                               /(N_hist-1))
    
    # Reset history sums
    score.sum_.fill(0.0)
    score.sum_sq.fill(0.0)

def tally_global_closeout():
    N_hist = mcdc.setting.N_hist
    i_iter = mcdc.i_iter

    # MPI reduce
    mcdc.tally_global.nuSigmaF = mpi.allreduce(mcdc.tally_global.nuSigmaF)
    if mcdc.setting.mode_alpha:
        mcdc.tally_global.inverse_speed = \
            mpi.allreduce(mcdc.tally_global.inverse_speed)
    
    # Update and store k_eff
    mcdc.tally_global.k_eff = mcdc.tally_global.nuSigmaF/N_hist
    mcdc.tally_global.k_iterate[i_iter] = mcdc.tally_global.k_eff
    
    # Update and store alpha_eff
    if mcdc.setting.mode_alpha:
        k_eff         = mcdc.tally_global.k_eff
        inverse_speed = mcdc.tally_global.inverse_speed/N_hist

        mcdc.tally_global.alpha_eff += (k_eff - 1.0)/inverse_speed
        mcdc.tally_global.alpha_iterate[i_iter] = mcdc.tally_global.alpha_eff
                
    # Reset accumulators
    mcdc.tally_global.nuSigmaF = 0.0
    if mcdc.setting.mode_alpha:
        mcdc.tally_global.inverse_speed = 0.0        

def manage_particle_banks():
    if mcdc.setting.mode_eigenvalue:
        # Normalize weight
        mpi.normalize_weight(mcdc.bank_census, mcdc.setting.N_hist)

    # Rebase RNG for population control
    mcdc.rng.skip_ahead_strides(mpi.work_size_total-mpi.work_start)
    mcdc.rng.rebase()

    # Population control
    mcdc_.population_control(mcdc.bank_census, mcdc.setting.N_hist, 
                             mcdc.bank_source)
    mcdc.rng.rebase()
    
    # Update MPI-related global variables
    mcdc.N_work = mpi.work_size
    mcdc.master = mpi.master

    # Zero out census bank
    mcdc.bank_census['size'] = 0

def generate_hdf5():
    msg = " Generating tally HDF5 files..."
    if mcdc.setting.progress_bar: msg = "\n" + msg
    print_msg(msg)

    # Save tallies to HDF5
    if mpi.master:
        with h5py.File(mcdc.setting.output_name+'.h5', 'w') as f:
            # Runtime
            f.create_dataset("runtime",data=np.array([mcdc.runtime_total]))
            f.create_dataset("runtime_pct",data=np.array([mcdc.runtime_pct]))

            # Tally
            T = mcdc.tally
            f.create_dataset("tally/grid/t", data=T.mesh.t())
            f.create_dataset("tally/grid/x", data=T.mesh.x())
            f.create_dataset("tally/grid/y", data=T.mesh.y())
            f.create_dataset("tally/grid/z", data=T.mesh.z())
            
            # Scores
            if T.flux:
                f.create_dataset("tally/flux/mean",
                                 data=np.squeeze(T.score_flux.mean))
                f.create_dataset("tally/flux/sdev",
                                 data=np.squeeze(T.score_flux.sdev))
                T.score_flux.mean.fill(0.0)
                T.score_flux.sdev.fill(0.0)
            if T.current:
                f.create_dataset("tally/current/mean",
                                 data=np.squeeze(T.score_current.mean))
                f.create_dataset("tally/current/sdev",
                                 data=np.squeeze(T.score_current.sdev))
                T.score_current.mean.fill(0.0)
                T.score_current.sdev.fill(0.0)
            if T.eddington:
                f.create_dataset("tally/eddington/mean",
                                 data=np.squeeze(T.score_eddington.mean))
                f.create_dataset("tally/eddington/sdev",
                                 data=np.squeeze(T.score_eddington.sdev))
                T.score_eddington.mean.fill(0.0)
                T.score_eddington.sdev.fill(0.0)
            if T.flux_x:
                f.create_dataset("tally/flux-x/mean",
                                 data=np.squeeze(T.score_flux_x.mean))
                f.create_dataset("tally/flux-x/sdev",
                                 data=np.squeeze(T.score_flux_x.sdev))
                T.score_flux_x.mean.fill(0.0)
                T.score_flux_x.sdev.fill(0.0)
            if T.flux_t:
                f.create_dataset("tally/flux-t/mean",
                                 data=np.squeeze(T.score_flux_t.mean))
                f.create_dataset("tally/flux-t/sdev",
                                 data=np.squeeze(T.score_flux_t.sdev))
                T.score_flux_t.mean.fill(0.0)
                T.score_flux_t.sdev.fill(0.0)
            if T.current_x:
                f.create_dataset("tally/current-x/mean",
                                 data=np.squeeze(T.score_current_x.mean))
                f.create_dataset("tally/current-x/sdev",
                                 data=np.squeeze(T.score_current_x.sdev))
                T.score_current_x.mean.fill(0.0)
                T.score_current_x.sdev.fill(0.0)
                
            # Eigenvalues
            if mcdc.setting.mode_eigenvalue:
                f.create_dataset("keff",data=mcdc.tally_global.k_iterate)
                mcdc.tally_global.k_iterate.fill(0.0)
                if mcdc.setting.mode_alpha:
                    f.create_dataset("alpha_eff",data=mcdc.tally_global.alpha_iterate)
                    mcdc.tally_global.alpha_iterate.fill(0.0)
