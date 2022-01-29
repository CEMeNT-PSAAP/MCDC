import sys
import h5py
import numpy as np

from math import floor

import mcdc.random
import mcdc.mpi

from mcdc.point        import Point
from mcdc.particle     import Particle
from mcdc.distribution import DistPointIsotropic
from mcdc.constant     import SMALL, VERY_SMALL,  LCG_SEED, LCG_STRIDE, INF,\
                              EVENT_COLLISION, EVENT_SURFACE, EVENT_CENSUS
from mcdc.random       import RandomLCG
from mcdc.pct          import *
from mcdc.misc         import binary_search
from mcdc.print        import print_banner


class Simulator:
    def __init__(self, cells=[], sources=[], tallies=[], N_hist = 0, 
                 speed=[], decay=[]):

        # Basic inputs
        #   TODO: run in batches
        self.cells   = cells   # list of mcdc.Cell
        self.sources = sources # list of mcdc.Source
        self.tallies = tallies # list of mcdc.Tally
        self.speed   = speed   # array of particle MG speeds (for univerasl use)
        self.decay   = decay   # array of precursor group decay constants
                               # (for universal use)
        self.N_hist  = int(N_hist) # number of histories (per iteration)

        # Output file
        self.output = "output" # .h5 output file name
        
        # RNG settings
        self.seed   = LCG_SEED
        self.stride = LCG_STRIDE

        # Eigenvalue mode settings (see self.set_kmode)
        #   TODO: alpha eigenvalue with delayed neutrons
        #   TODO: shannon entropy
        self.mode_eigenvalue = False
        self.k_eff           = 1.0   # k effective that affects simulation
        self.N_iter          = 1     # number of iterations
        self.i_iter          = 0     # current iteration index
        self.mode_alpha      = False
        self.alpha_eff       = 0.0   # k effective that affects simulation
        
        # Population control settings (see self.set_pct)
        self.pct         = PCT_CO()
        self.census_time = [INF]

        # Particle banks
        #   TODO: use fixed memory allocations with helper indices
        self.bank_stored  = [] # for the next source loop
        self.bank_source  = [] # for current source loop
        self.bank_history = [] # for current history loop
        self.bank_fission = None # will point to one of the banks

        # Timer
        self.time_total = 0.0

        # Misc.
        self.isotropic_dir = DistPointIsotropic() # (see distribution.py)
        self.parallel_hdf5 = False # TODO
            

    def set_kmode(self, N_iter=1, k_init=1.0, alpha_mode=False, alpha_init=0.0):
        self.mode_eigenvalue = True
        self.N_iter          = N_iter
        self.k_eff           = k_init
        self.mode_alpha      = alpha_mode
        self.alpha_eff       = alpha_init

        # Mode-specific tallies
        # Accumulators
        self.nuSigmaF_sum = 0.0
        if self.mode_alpha:
            self.ispeed_sum = 0.0
        # MPI buffer
        self.nuSigmaF_buff = np.array([0.0])
        if self.mode_alpha:
            self.ispeed_buff = np.array([0.0])
        # Eigenvalue solution iterate
        if mcdc.mpi.master:
            self.k_mean = np.zeros(self.N_iter)
            if self.mode_alpha:
                self.alpha_mean = np.zeros(self.N_iter)

    def set_pct(self, pct='CO', census_time=[INF]):
        # Set technique
        if pct == 'None':
            self.pct = PCT_NONE()
        elif pct == 'SS':
            self.pct = PCT_SS()
        elif pct == 'SR':
            self.pct = PCT_SR()
        elif pct == 'CO':
            self.pct = PCT_CO()
        elif pct == 'COX':
            self.pct = PCT_COX()
        elif pct == 'DD':
            self.pct = PCT_DD()
        else:
            print("ERROR: Unknown PCT "+pct)
            sys.exit()

        # Set census time
        self.census_time = census_time

    # =========================================================================
    # Run ("main") -- SIMULATION LOOP
    # =========================================================================
    
    def run(self):
        print_banner()
        if mcdc.mpi.master:
            print(" Now running TNT...")
            sys.stdout.flush()

        # Start timer
        self.time_pct   = 0.0
        self.time_total = mcdc.mpi.Wtime()

        # Set group universal speed and decay constants if given
        if len(self.speed) > 0:
            for c in self.cells:
                c.material.speed = self.speed
        if len(self.decay) > 0:
            for c in self.cells:
                c.material.decay = self.decay

        # Set tally bins
        for T in self.tallies:
            T.setup_bins(self.N_iter) # Allocate tally bins (see tally.py)

        # Set pct
        self.pct.prepare(self.N_hist)
   
        # Setup RNG
        mcdc.random.rng = RandomLCG(seed=self.seed, stride=self.stride)

        # Normalize sources
        norm = 0.0
        for s in self.sources: norm += s.prob
        for s in self.sources: s.prob /= norm

        # Make sure no census time in eigenvalue mode
        if self.mode_eigenvalue:
            self.census_time = [INF]

        # Distribute work to processors
        mcdc.mpi.distribute_work(self.N_hist)

        # SIMULATION LOOP
        simulation_end = False
        while not simulation_end:
            # To which bank fission neutrons are stored?
            if self.mode_eigenvalue:
                self.bank_fission = self.bank_stored
            else:
                self.bank_fission = self.bank_history
           
            # SOURCE LOOP
            self.loop_source()
            
            # Tally closeout for eigenvalue mode
            if self.mode_eigenvalue:
                for T in self.tallies:
                    T.closeout(self.N_hist, self.i_iter)

                # MPI Reduce nuSigmaF
                mcdc.mpi.allreduce(self.nuSigmaF_sum, self.nuSigmaF_buff)
                if self.mode_alpha:
                    mcdc.mpi.allreduce(self.ispeed_sum, self.ispeed_buff)
                
                # Update keff
                self.k_eff = self.nuSigmaF_buff[0]/self.N_hist
                if mcdc.mpi.master:
                    self.k_mean[self.i_iter] = self.k_eff
                
                if self.mode_alpha:
                    self.alpha_eff += (self.k_eff - 1.0)/(self.ispeed_buff[0]/self.N_hist)
                    if mcdc.mpi.master:
                        self.alpha_mean[self.i_iter] = self.alpha_eff
                            
                # Reset accumulator
                self.nuSigmaF_sum = 0.0
                if self.mode_alpha:
                    self.ispeed_sum = 0.0
            
                # Progress printout
                #   TODO: print in table format 
                if mcdc.mpi.master:
                    sys.stdout.write('\r')
                    sys.stdout.write("\033[K")
                    if not self.mode_alpha:
                        print(" %-4i %.5f"%(self.i_iter+1,self.k_eff))
                    else:
                        print(" %-4i %.5f %.3e"%
                                (self.i_iter+1,self.k_eff,self.alpha_eff))
                    sys.stdout.flush()

            # Simulation end?
            if self.mode_eigenvalue:
                self.i_iter += 1
                if self.i_iter == self.N_iter: simulation_end = True
            elif not self.bank_stored: simulation_end = True

            # Manage particle banks
            if not simulation_end:
                if self.mode_eigenvalue:
                    # Normalize weight
                    mcdc.mpi.normalize_weight(self.bank_stored, self.N_hist)
                tot = 0.0
                for P in self.bank_stored:
                    tot += P.wgt

                # Rebase RNG for population control
                mcdc.random.rng.skip_ahead(
                    mcdc.mpi.work_size_total-mcdc.mpi.work_start, rebase=True)

                # Population control
                click = mcdc.mpi.Wtime()
                self.bank_stored = self.pct(self.bank_stored, self.N_hist)
                self.time_pct += mcdc.mpi.Wtime() - click
                
                # Set stored bank as source bank for the next iteration
                self.bank_source = self.bank_stored
                self.bank_stored = []
            else:
                self.bank_source = []
                self.bank_stored = []
                
        # Tally closeout for fixed-source mode
        if not self.mode_eigenvalue:
            for T in self.tallies:
                T.closeout(self.N_hist, 0)

        # Stop timer
        self.time_total = mcdc.mpi.Wtime() - self.time_total

        # Save tallies to HDF5
        if mcdc.mpi.master and self.tallies:
            with h5py.File(self.output+'.h5', 'w') as f:
                # Runtime
                f.create_dataset("runtime",data=np.array([self.time_total]))
                f.create_dataset("runtime_pct",data=np.array([self.time_pct]))

                # Tallies
                for T in self.tallies:
                    if T.filter_flag_energy:
                    # Filters
                        f.create_dataset(T.name+"/energy_grid", 
                                         data=T.filter_energy.grid)
                    if T.filter_flag_angular:
                        f.create_dataset(T.name+"/angular_grid",
                                         data=T.filter_angular.grid)
                    if T.filter_flag_time:
                        f.create_dataset(T.name+"/time_grid",
                                         data=T.filter_time.grid)
                    if T.filter_flag_spatial:
                        f.create_dataset(T.name+"/spatial_grid",
                                         data=T.filter_spatial.grid)
                    
                    # Scores
                    for S in T.scores:
                        f.create_dataset(T.name+"/"+S.name+"/mean",
                                         data=np.squeeze(S.mean))
                        f.create_dataset(T.name+"/"+S.name+"/sdev",
                                         data=np.squeeze(S.sdev))
                        S.mean.fill(0.0)
                        S.sdev.fill(0.0)
                    
                # Eigenvalues
                if self.mode_eigenvalue:
                    f.create_dataset("keff",data=self.k_mean)
                    self.k_mean.fill(0.0)
                    if self.mode_alpha:
                        f.create_dataset("alpha_eff",data=self.alpha_mean)
                        self.alpha_mean.fill(0.0)

        if mcdc.mpi.master:
            print('\n')
            sys.stdout.flush()

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
                # Sample source
                xi = mcdc.random.rng()
                tot = 0.0
                source = None
                for s in self.sources:
                    tot += s.prob
                    if xi < tot:
                        source = s
                        break
                P = source.get_particle()

                # Set cell if not given
                if not P.cell: 
                    self.set_cell(P)
                # Set time_idx if not given
                if not P.time_idx:
                    self.set_time_idx(P)

                self.bank_history.append(P)
            else:
                self.bank_history.append(self.bank_source[i])
            
            # History loop
            self.loop_history()
            
            # Progress printout
            # TODO: Does not seem to work with MPI
            #if mcdc.mpi.master and not self.mode_eigenvalue:
            if mcdc.mpi.master:
                perc = (i+1.0)/mcdc.mpi.work_size
                sys.stdout.write('\r')
                sys.stdout.write(" [%-28s] %d%%" % ('='*int(perc*28), perc*100.0))
                sys.stdout.flush()

    
    def set_cell(self, P):
        pos = P.pos
        t = P.time
        C = None
        for cell in self.cells:
            if cell.test_point(pos,t):
                C = cell
                break
        if C == None:
            print("ERROR: A particle is lost at "+str(pos))
            sys.exit()

        P.cell = C
        
    def set_time_idx(self, P):
        t = P.time
        idx = binary_search(t, self.census_time) + 1

        if idx == len(self.census_time):
            P.alive = False
            idx = None
        elif P.time == self.census_time[idx]:
            idx += 1
        P.time_idx = idx


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
            P.speed = P.cell.material.speed[P.g]
        
            # =================================================================
            # Get distances to events
            # =================================================================
    
            # Distance to collision
            d_coll = self.get_collision_distance(P)

            # Nearest surface and distance to hit
            S, d_surf = self.surface_distance(P)

            # Distance to census
            t_census = self.census_time[P.time_idx]
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
                P.time_idx += 1
                # Not final census?
                if P.time_idx < len(self.census_time):
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
                wgt    = P.wgt_old
                SigmaF = P.cell_old.material.fission[P.g_old]
                
                # nu
                nu = P.cell_old.material.nu_p[P.g_old]
                if P.cell_old.material.J > 0:
                    nu_d = P.cell.material.nu_d[P.g]
                    nu += sum(nu_d)

                nuSigmaF = nu*SigmaF
                self.nuSigmaF_sum += wgt*P.distance*nuSigmaF

                if self.mode_alpha:
                    self.ispeed_sum += wgt*P.distance/P.cell.material.speed[P.g_old]
            
            # =================================================================
            # Closeout
            # =================================================================    

            # Reset particle record
            P.reset_record()

    # =========================================================================
    # Particle transports
    # =========================================================================

    def get_collision_distance(self, P):
        xi     = mcdc.random.rng()
        SigmaT = P.cell.material.total[P.g]

        SigmaT += VERY_SMALL # To ensure non-zero value

        if self.mode_alpha:
            SigmaT += abs(self.alpha_eff)/P.cell.material.speed[P.g]

        d_coll  = -np.log(xi)/SigmaT
        return d_coll

    # Get nearest surface and distance to hit for particle P
    def surface_distance(self,P):
        S     = None
        d_surf = np.inf
        for surf in P.cell.surfaces:
            d = surf.distance(P.pos, P.dir, P.time, P.speed)
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
        SigmaT = P.cell.material.total[P.g]
        SigmaC = P.cell.material.capture[P.g]
        SigmaS = P.cell.material.scatter[P.g]
        SigmaF = P.cell.material.fission[P.g]

        if self.mode_alpha:
            SigmaT += abs(self.alpha_eff)/P.cell.material.speed[P.g]
        
        # Sample and then implement reaction type
        xi = mcdc.random.rng()*SigmaT
        tot = SigmaS
        if tot > xi:
            # Scattering
            self.collision_scattering(P)
        else:
            tot += SigmaF
            if tot > xi:
                # Fission
                self.collision_fission(P)
            else:
                tot += SigmaC
                if tot > xi:
                    # Capture
                    self.collision_capture(P)
                else:
                    # Time-correction
                    if self.alpha_eff > 0:
                        P.alive = False
                    else:
                        P_out = Particle(P.pos, P.dir, P.g, P.time, P.wgt, P.cell, P.time_idx)
                        self.bank_history.append(P_out)
    
    def collision_capture(self, P):
        P.alive = False
        
    def collision_scattering(self, P):
        # Ger outgoing spectrum
        chi_s = P.cell.material.chi_s[P.g]
        G     = P.cell.material.G
        
        # Sample outgoing energy
        xi  = mcdc.random.rng()
        tot = 0.0
        for g_out in range(G):
            tot += chi_s[g_out]
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
            
        P.dir = dir_final
        
    def collision_fission(self, P):
        # Kill the current particle
        P.alive = False

        # Get group numbers
        G = P.cell.material.G
        J = P.cell.material.J

        # Total nu
        nu_p = P.cell.material.nu_p[P.g]
        nu = nu_p
        if J>0: 
            nu_d = P.cell.material.nu_d[P.g]
            nu += sum(nu_d)

        # Sample number of fission neutrons
        #   in fixed-source, k_eff = 1.0
        N = floor(nu/self.k_eff + mcdc.random.rng())

        # Push fission neutrons to bank
        for n in range(N):
            # Determine if it's prompt or delayed neutrons, 
            # then get the energy spectrum and decay constant
            xi  = mcdc.random.rng()*nu
            tot = nu_p
            # Prompt?
            if tot > xi:
                spectrum = P.cell.material.chi_p[P.g]
                decay    = INF
            else:
                # Which delayed group?
                for j in range(J):
                    tot += nu_d[j]
                    if tot > xi:
                        spectrum = P.cell.material.chi_d[j]
                        decay    = P.cell.material.decay[j]
                        break

            # Sample emission time
            xi = mcdc.random.rng()
            t_out = P.time - np.log(xi)/decay

            # Skip if it's beyound final census time
            if t_out > self.census_time[-1]:
                continue

            # Sample outgoing energy
            xi  = mcdc.random.rng()
            tot = 0.0
            for g_out in range(G):
                tot += spectrum[g_out]
                if tot > xi:
                    break

            # Sample isotropic direction
            dir = self.isotropic_dir.sample()
           
            # Create the outgoing particle
            P_out = Particle(P.pos, dir, g_out, t_out, P.wgt, P.cell, P.time_idx)
            self.set_time_idx(P_out)

            # Bank
            self.bank_fission.append(P_out)
