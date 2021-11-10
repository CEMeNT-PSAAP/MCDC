import sys
import h5py
import numpy as np

from math import floor

import mcdc.random
import mcdc.mpi
import mcdc.vrt

from mcdc.point        import Point
from mcdc.particle     import Particle
from mcdc.distribution import DistPointIsotropic
from mcdc.constant     import SMALL, VERY_SMALL,  LCG_SEED, LCG_STRIDE, INF,\
                              EVENT_COLLISION, EVENT_SURFACE, EVENT_CENSUS
from mcdc.random       import RandomLCG
from mcdc.pct          import *
from mcdc.misc         import binary_search


class Simulator:
    def __init__(self, speeds, cells, source, tallies = [], N_hist = 1,
                 seed = LCG_SEED, stride = LCG_STRIDE):

        # Basic settings
        self.speeds  = speeds   # array of particle MG speeds
        self.cells   = cells    # list of Cells (see geometry.py)
        self.source  = source   # Source (see particle.py)
        self.N_hist  = N_hist   # number of histories
        self.tallies = tallies  # list of Tallies (see tally.py)
        self.output  = "output" # .h5 output file name
        
        # RNG settings
        self.seed   = seed
        self.stride = stride
        
        # Fixed-source mode
        self.mode_fixed_source = True # fixed-source flag

        # Eigenvalue settings (see self.k_mode)
        #   TODO: alpha eigenvalue
        #   TODO: shannon entropy
        self.mode_k          = False # k-eigenvalue flag
        self.mode_eigenvalue = False # eigenvalue flag
        self.N_iter          = 1     # number of iterations
        self.i_iter          = 0     # current iteration index
        self.k_eff           = 1.0   # k effective that affects simulation
        
        # Population control settings (see self.set_pct)
        #   TODO: census
        self.pct     = PCT_SSU()
        self.census_time = [INF]

        # Variance Reduction Technique settings
        #   TODO: implicit fission production
        self.mode_analog     = True
        self.vrt_capture     = False
        self.vrt_fission     = False
        self.vrt_wgt_roulette  = 0.0
        self.vrt_wgt_survive = 1.0
        
        # Particle banks
        #   TODO: use fixed memory allocations with helper indices
        self.bank_stored  = [] # for the next source loop
        self.bank_source  = [] # for current source loop
        self.bank_history = [] # for current history loop
        self.bank_fission = None # will point to one of the banks
        
        # Misc.
        self.isotropic_dir = DistPointIsotropic() # (see distribution.py)
        self.parallel_hdf5 = False # TODO
            

    def set_kmode(self, N_iter=1, k_init=1.0):
        self.mode_k          = True
        self.mode_eigenvalue = True
        self.N_iter          = N_iter
        self.k_eff           = k_init

        self.mode_fixed_source = False

        # Mode-specific tallies
        # Accumulators
        self.nuSigmaF_sum = 0.0
        # MPI buffer
        self.nuSigmaF_buff = np.array([0.0])
        # Iteration solution for k
        if mcdc.mpi.master:
            self.k_mean = np.zeros(self.N_iter)

    def set_pct(self, pct='SS-U', census_time=[INF]):
        # Set technique
        if pct == 'SS-U':
            pass
        elif pct == 'SR-U':
            self.pct = PCT_SRU()
        elif pct == 'CO-U':
            self.pct = PCT_COU()
        elif pct == 'COx-U':
            self.pct = PCT_COxU()
        elif pct == 'DD-U':
            self.pct = PCT_DDU()
        elif pct == 'DD-Uori':
            self.pct = PCT_DDUOri()
        elif pct == 'SS-W':
            self.pct = PCT_SSW()
        elif pct == 'SR-W':
            self.pct = PCT_SRW()
        elif pct == 'CO-W':
            self.pct = PCT_COW()
        else:
            print("ERROR: Unknown PCT "+pct)
            sys.exit()

        # Set census time
        self.census_time = census_time

    def set_vrt(self, continuous_capture=False, implicit_fission=False,
                wgt_roulette=0.0, wgt_survive=1.0):
        self.vrt_capture = continuous_capture
        self.vrt_fission = implicit_fission
        self.vrt_wgt_roulette  = wgt_roulette
        self.vrt_wgt_survive = wgt_survive


    # =========================================================================
    # Run ("main") --> SIMULATION LOOP
    # =========================================================================
    
    def run(self):
        # Start timer
        time = mcdc.mpi.Wtime()

        # Set tally bins
        for T in self.tallies:
            T.setup_bins(self.N_iter) # Allocate tally bins (see tally.py)
   
        # Setup RNG
        mcdc.random.rng = RandomLCG(seed=self.seed, stride=self.stride)

        # Setup VRT
        mcdc.vrt.capture      = self.vrt_capture
        mcdc.vrt.fission      = self.vrt_fission
        mcdc.vrt.wgt_roulette = self.vrt_wgt_roulette
        mcdc.vrt.wgt_survive  = self.vrt_wgt_survive

        # Distribute work to processors
        mcdc.mpi.distribute_work(self.N_hist)

        simulation_end = False
        while not simulation_end:
            # To which bank fission neutrons are stored?
            if self.mode_fixed_source:
                self.bank_fission = self.bank_history
            if self.mode_eigenvalue:
                self.bank_fission = self.bank_stored
           
            # Source loop
            self.loop_source()
            
            # Closeout
            if self.mode_eigenvalue: 
                simulation_end = self.closeout_eigenvalue_iteration()
            elif self.mode_fixed_source:
                simulation_end = self.closeout_fixed_source()

        # Stop timer
        time = mcdc.mpi.Wtime() - time

        # Simulation closeout
        self.closeout_simulation(time)

    def closeout_eigenvalue_iteration(self):    
        # Tally source closeout
        for T in self.tallies:
            T.closeout(self.N_hist, self.i_iter)

        # MPI Reduce nuSigmaF
        mcdc.mpi.allreduce(self.nuSigmaF_sum, self.nuSigmaF_buff)
        
        # Update keff
        self.k_eff = self.nuSigmaF_buff[0]/mcdc.mpi.work_size_total
        if mcdc.mpi.master:
            self.k_mean[self.i_iter] = self.k_eff
                    
        # Reset accumulator
        self.nuSigmaF_sum = 0.0

        # Next iteration?
        self.i_iter += 1           
        if self.i_iter < self.N_iter:
            simulation_end = False

            # Normalize weight
            mcdc.mpi.normalize_weight(self.bank_stored, self.N_hist)

            # Rebase RNG for population control
            mcdc.random.rng.skip_ahead(mcdc.mpi.work_size_total-mcdc.mpi.work_start,
                                       rebase=True)

            # Population control stored bank
            self.bank_stored = self.pct(self.bank_stored, self.N_hist)

            # Set stored bank as source bank for the next iteration
            self.bank_source = self.bank_stored
            self.bank_stored = []

        else:
            self.bank_source = []
            self.bank_stored = []
            simulation_end = True
        
        # Progress printout
        #   TODO: make optional. print in a table format
        
        if mcdc.mpi.master:
            print(self.i_iter,self.k_eff)
            sys.stdout.flush()
        
        return simulation_end 

    def closeout_fixed_source(self):    
        if self.bank_stored:
            simulation_end = False

            # Rebase RNG for population control
            mcdc.random.rng.skip_ahead(mcdc.mpi.work_size_total-mcdc.mpi.work_start,
                                       rebase=True)

            # Population control stored bank
            self.bank_stored = self.pct(self.bank_stored, self.N_hist)

            # Set stored bank as source bank for the next iteration
            self.bank_source = self.bank_stored
            self.bank_stored = []

        else:
            self.bank_source = []
            self.bank_stored = []
            simulation_end = True
            
            # Tally closeout
            for T in self.tallies:
                T.closeout(self.N_hist, 0)

        return simulation_end

    def closeout_simulation(self, time):
        # =========================================================================
        # Save tallies to HDF5
        # =========================================================================

        #with h5py.File(self.output+'.h5', 'w', driver='mpio', comm=mcdc.mpi.comm) as f:
        if mcdc.mpi.master and self.tallies:
            with h5py.File(self.output+'.h5', 'w') as f:
                f.create_dataset("runtime",data=np.array([time]))
                # Tallies
                for T in self.tallies:
                    if T.filter_flag_energy:
                        f.create_dataset(T.name+"/energy_grid",data=T.filter_energy.grid)
                    if T.filter_flag_angular:
                        f.create_dataset(T.name+"/angular_grid",data=T.filter_angular.grid)
                    if T.filter_flag_time:
                        f.create_dataset(T.name+"/time_grid",data=T.filter_time.grid)
                    if T.filter_flag_spatial:
                        f.create_dataset(T.name+"/spatial_grid",data=T.filter_spatial.grid)
                    
                    # Scores
                    for S in T.scores:
                        f.create_dataset(T.name+"/"+S.name+"/mean",data=np.squeeze(S.mean))
                        f.create_dataset(T.name+"/"+S.name+"/sdev",data=np.squeeze(S.sdev))
                        S.mean.fill(0.0)
                        S.sdev.fill(0.0)
                    
                # Eigenvalues
                if self.mode_eigenvalue:
                    f.create_dataset("keff",data=self.k_mean)
                    self.k_mean.fill(0.0)

        # Reset simulation parameters
        self.i_iter = 0


    # =========================================================================
    # SOURCE LOOP
    # =========================================================================
    
    def loop_source(self):
        # Rebase rng skip_ahead seed
        mcdc.random.rng.skip_ahead(mcdc.mpi.work_start, rebase=True)

        # Loop over sources
        for i in range(mcdc.mpi.work_size):
            # Initialize RNG wrt global index
            mcdc.random.rng.skip_ahead(i)

            # Get a source particle and put into history bank
            if not self.bank_source:
                # Initial source
                P = self.source.get_particle()

                # Set cell if not given
                if not P.cell: 
                    self.set_cell(P)
                # Set census_idx if not given
                if not P.census_idx:
                    self.set_census_idx(P)

                self.bank_history.append(P)
            else:
                self.bank_history.append(self.bank_source[i])
            
            # History loop
            self.loop_history()
            
            # Super rough estimate of progress
            #   TODO: Make it optional. A progress bar?
            '''
            if mcdc.mpi.master and self.mode_fixed_source:
                prog = (i+1)/mcdc.mpi.work_size*100
                print('%.2f'%prog,'%')
                sys.stdout.flush()
            '''

    def set_cell(self, P):
        pos = P.pos
        C = None
        for cell in self.cells:
            if cell.test_point(pos):
                C = cell
                break
        if C == None:
            print("ERROR: A particle is lost at "+str(pos))
            sys.exit()

        P.cell = C
        
    def set_census_idx(self, P):
        t = P.time
        idx = binary_search(t, self.census_time) + 1

        if idx == len(self.census_time):
            P.alive = False
            idx = None
        elif P.time == self.census_time[idx]:
            idx += 1
        P.census_idx = idx

    # =========================================================================
    # HISTORY LOOP
    # =========================================================================
    
    def loop_history(self):
        while self.bank_history:
            # Get particle from history bank
            P = self.bank_history.pop()
            
            # Particle loop
            self.loop_particle(P)

        # Tally history closeout
        for T in self.tallies:
            T.closeout_history()
            
    # =========================================================================
    # PARTICLE LOOP
    # =========================================================================
    
    def loop_particle(self, P):
        while P.alive:
            # =================================================================
            # Setup
            # =================================================================
    
            # Record initial parameters
            P.save_previous_state()

            # Get speed and XS (not neeeded for MG mode)
            P.speed = self.speeds[P.g]
        
            # =================================================================
            # Get distances to events
            # =================================================================
    
            # Distance to collision
            d_coll = self.get_collision_distance(P)

            # Nearest surface and distance to hit
            S, d_surf = self.surface_distance(P)

            # Distance to census
            t_census = self.census_time[P.census_idx]
            d_census = P.speed*(t_census - P.time)

            # =================================================================
            # Choose event
            # =================================================================
    
            # Collision, surface hit, or census?
            event  = EVENT_COLLISION
            d_move = d_coll
            if d_move > d_surf:
                event  = EVENT_SURFACE
                d_move = d_surf
            if d_move > d_census:
                event  = EVENT_CENSUS
                d_move = d_census
            
            # =================================================================
            # Move to event
            # =================================================================
    
            # Move particle
            self.move_particle(P, d_move)

            # Continuous capture?
            if mcdc.vrt.capture:
                SigmaC  = P.cell.material.SigmaC[P.g]
                P.wgt  *= np.exp(-d_move*SigmaC)

            # =================================================================
            # Perform event
            # =================================================================    

            if event == EVENT_COLLISION:
                # Sample collision
                self.collision(P)

            elif event == EVENT_SURFACE:
                # Record surface hit
                P.surface = S

                # Implement surface hit
                self.surface_hit(P)
            
            elif event == EVENT_CENSUS:
                # Cross the time boundary
                d = SMALL*P.speed
                self.move_particle(P, d)

                # Increment index
                P.census_idx += 1
                # Not final census?
                if P.census_idx < len(self.census_time):
                    # Store for next time census
                    self.bank_stored.append(P.create_copy())
                P.alive = False

            # =================================================================
            # Scores
            # =================================================================    

            # Score tallies
            for T in self.tallies:
                T.score(P)
                
            # Score eigenvalue tallies
            if self.mode_eigenvalue:
                wgt = P.wgt_old
                # Continuous capture?
                if mcdc.vrt.capture:
                    SigmaC  = P.cell_old.material.SigmaC[P.g_old]
                    wgt     = (P.wgt_old-P.wgt)/(P.distance*SigmaC)

                nu       = P.cell_old.material.nu[P.g_old]
                SigmaF   = P.cell_old.material.SigmaF[P.g_old]
                nuSigmaF = nu*SigmaF
                self.nuSigmaF_sum += wgt*P.distance*nuSigmaF
            
            # =================================================================
            # Closeout
            # =================================================================    

            # Reset particle record
            P.reset_record()

            # Cutoff?
            if P.alive and P.wgt <= mcdc.vrt.wgt_roulette:
                # Russian-roulette
                p_survive = P.wgt/mcdc.vrt.wgt_survive
                xi = mcdc.random.rng()
                if xi < p_survive:
                    # Survive
                    P.wgt = mcdc.vrt.wgt_survive
                else:
                    # Terminate
                    P.alive = False
            
    # =========================================================================
    # Particle transports
    # =========================================================================

    def get_collision_distance(self, P):
        xi     = mcdc.random.rng()
        SigmaT = P.cell.material.SigmaT[P.g]

        # Continuous capture?
        if mcdc.vrt.capture:
            SigmaC  = P.cell.material.SigmaC[P.g]
            SigmaT -= SigmaC
        
        SigmaT += VERY_SMALL # To ensure non-zero value
        d_coll  = -np.log(xi)/SigmaT
        return d_coll

    # Get nearest surface and distance to hit for particle P
    def surface_distance(self,P):
        S     = None
        d_surf = np.inf
        for surf in P.cell.surfaces:
            d = surf.distance(P.pos, P.dir)
            if d < d_surf:
                S     = surf;
                d_surf = d;
        return S, d_surf

    def move_particle(self, P, d):
        # 4D Move
        P.pos  += P.dir*d
        P.time += d/P.speed
        
        # Record distance traveled
        P.distance += d
        
    def surface_hit(self, P):
        # Implement BC
        P.surface.bc(P)

        # Small kick (see constant.py) to make sure crossing
        self.move_particle(P, SMALL)
        
        # Set new cell
        if P.alive:
            self.set_cell(P)
            
    def collision(self, P):
        SigmaT = P.cell.material.SigmaT[P.g]
        SigmaC = P.cell.material.SigmaC[P.g]
        SigmaS = P.cell.material.SigmaS[P.g]
        SigmaF = P.cell.material.SigmaF[P.g]
        
        # Continuous capture?
        if mcdc.vrt.capture:
            SigmaT -= SigmaC

        # Implicit fission?
        if mcdc.vrt.fission:
            # Forced fission
            self.collision_fission(P)
            
            # Revive the current particle and set the post-collision weight
            P.alive    = True
            P.wgt_post = P.wgt*(SigmaT-SigmaF)/SigmaT
            
            # Modify SigmaT and SigmaF 
            # to make the following reaction type sampling consistent
            SigmaT -= SigmaF
            SigmaF  = 0.0

        # Scattering or absorption?
        xi = mcdc.random.rng()*SigmaT
        if SigmaS > xi:
            # Scattering
            self.collision_scattering(P)
        else:
            # Fission or capture?
            if SigmaS + SigmaF > xi:
                # Fission
                self.collision_fission(P)
            else:
                # Capture
                self.collision_capture(P)
    
    def collision_capture(self, P):
        P.alive = False
        
    def collision_scattering(self, P):
        SigmaS_diff = P.cell.material.SigmaS_diff[P.g]
        SigmaS      = P.cell.material.SigmaS[P.g]
        G           = len(SigmaS_diff)
        
        # Sample outgoing energy
        xi  = mcdc.random.rng()*SigmaS
        tot = 0.0
        for g_out in range(G):
            tot += SigmaS_diff[g_out]
            if tot > xi:
                break
        P.g = g_out
        
        # Sample scattering angle
        mu0 = 2.0*mcdc.random.rng() - 1.0;
        
        # Scatter particle
        self.scatter(P,mu0)

    # Scatter direction with scattering cosine mu
    def scatter(self, P, mu):
        # Sample azimuthal direction
        azi     = 2.0*np.pi*mcdc.random.rng()
        cos_azi = np.cos(azi)
        sin_azi = np.sin(azi)
        Ac      = (1.0 - mu**2)**0.5
        
        dir_final = Point(0,0,0)
    
        if P.dir.z != 1.0:
            B = (1.0 - P.dir.z**2)**0.5
            C = Ac/B
            
            dir_final.x = P.dir.x*mu + (P.dir.x*P.dir.z*cos_azi - P.dir.y*sin_azi)*C
            dir_final.y = P.dir.y*mu + (P.dir.y*P.dir.z*cos_azi + P.dir.x*sin_azi)*C
            dir_final.z = P.dir.z*mu - cos_azi*Ac*B
        
        # If dir = 0i + 0j + k, interchange z and y in the scattering formula
        else:
            B = (1.0 - P.dir.y**2)**0.5
            C = Ac/B
            
            dir_final.x = P.dir.x*mu + (P.dir.x*P.dir.y*cos_azi - P.dir.z*sin_azi)*C
            dir_final.z = P.dir.z*mu + (P.dir.z*P.dir.y*cos_azi + P.dir.x*sin_azi)*C
            dir_final.y = P.dir.y*mu - cos_azi*Ac*B
            
        P.dir.copy(dir_final)        
        
    def collision_fission(self, P):
        # Kill the current particle
        P.alive = False
        
        SigmaF_diff = P.cell.material.SigmaF_diff[P.g]
        SigmaF      = P.cell.material.SigmaF[P.g]
        nu          = P.cell.material.nu[P.g]
        G           = len(SigmaF_diff)

        # Implicit fission?
        iFission = 1.0 # Multiplying factor
        if mcdc.vrt.fission:
            SigmaT = P.cell.material.SigmaT[P.g]
            SigmaC = P.cell.material.SigmaC[P.g]
            if mcdc.vrt.capture:
                SigmaT -= SigmaC
            iFission = SigmaF/SigmaT
        
        # Set fission neutron weight and effective nu
        if mcdc.vrt.wgt_roulette == 0.0:
            wgt    = P.wgt*iFission
            nu_eff = nu
        else:
            wgt    = mcdc.vrt.wgt_survive
            nu_eff = P.wgt/wgt*nu*iFission

        # Sample number of fission neutrons
        #   in fixed-source, k_eff = 1.0
        N = floor(nu_eff/self.k_eff + mcdc.random.rng())

        # Push fission neutrons to bank
        for n in range(N):
            # Sample outgoing energy
            xi  = mcdc.random.rng()*SigmaF
            tot = 0.0
            for g_out in range(G):
                tot += SigmaF_diff[g_out]
                if tot > xi:
                    break
            
            # Sample isotropic direction
            dir = self.isotropic_dir.sample()
            
            # Bank
            self.bank_fission.append(Particle(P.pos, dir, g_out, P.time, wgt, 
                                              P.cell, P.census_idx))
