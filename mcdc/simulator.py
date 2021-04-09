import sys
import h5py
import numpy as np

from math import floor

import mcdc.random
import mcdc.mpi

from mcdc.point        import Point
from mcdc.particle     import Particle
from mcdc.distribution import DistPointIsotropic
from mcdc.constant     import SMALL_KICK, LCG_SEED, LCG_STRIDE
from mcdc.random       import RandomLCG
from mcdc.popctrl      import PopCtrlSimple, PopCtrlSR, PopCtrlComb


class Simulator:
    def __init__(self, speeds, cells, source, N_hist = 1, tallies = [],
                 k_mode = False, k_init = 1.0, N_iter = 1,
                 output = 'output', population_control='simple',
                 seed = LCG_SEED, stride = LCG_STRIDE):

        # Basic settings
        self.speeds  = speeds  # array of particle MG speeds
        self.cells   = cells   # list of Cells (see geometry.py)
        self.source  = source  # Source (see particle.py)
        self.N_hist  = N_hist  # number of histories
        self.tallies = tallies # list of Tallies (see tally.py)
        self.output  = output  # .h5 output file name
        
        # Eigenvalue settings
        #   TODO: alpha eigenvalue
        #   TODO: shannon entropy
        self.mode_k          = k_mode # k-eigenvalue flag
        self.mode_eigenvalue = k_mode # eigenvalue flag
        self.N_iter          = N_iter # number of iterations
        self.i_iter          = 0      # current iteration index
        self.k_eff           = k_init # k effective that affects simulation
        
        # Fixed-source mode
        self.mode_fixed_source = not self.mode_eigenvalue # fixed-source flag

        # Particle banks
        #   TODO: use fixed memory allocations with helper indices
        self.bank_stored  = [] # for the next source loop
        self.bank_source  = [] # for current source loop
        self.bank_history = [] # for current history loop
        self.bank_fission = None # will point to one of the banks
        
        # RNG settings
        self.seed   = seed
        self.stride = stride
        
        # Variance Reduction Technique settings
        #   TODO: implicit capture, implicit fission production
        self.mode_analog = True
        
        # Population control
        #   TODO: Inner and outer census
        if population_control == 'simple':
            self.popctrl = PopCtrlSimple()
        elif population_control == 'split-roulette':
            self.popctrl = PopCtrlSR()
        elif population_control == 'comb':
            self.popctrl = PopCtrlComb()
        # Check if the chosen population control is appropriate
        if population_control == 'simple' and not self.mode_analog:
            print("ERROR: Population control Simple is only applicable for analog simulation ")
            sys.exit()

        # Misc.
        self.isotropic_dir = DistPointIsotropic() # (see distribution.py)
            
        # Eigenvalue-specific tallies
        if self.mode_eigenvalue:
            # Accumulators
            self.nuSigmaF_sum = 0.0
            # MPI buffer
            self.nuSigmaF_buff = np.array([0.0])
            # Iteration solution for k
            if mcdc.mpi.master:
                self.k_mean = np.zeros(self.N_iter)
            
        # Set tally bins
        for T in tallies:
            T.setup_bins(N_iter) # Allocate tally bins (see tally.py)
   

    # =========================================================================
    # Run ("main") --> SIMULATION LOOP
    # =========================================================================
    
    def run(self):
        # Setup RNG
        mcdc.random.rng = RandomLCG(seed=self.seed, stride=self.stride)

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
            
            # Increment iteration index
            self.i_iter += 1           
            
            # Closeout
            if self.mode_eigenvalue: 
                simulation_end = self.closeout_eigenvalue_iteration()
            elif self.mode_fixed_source:
                simulation_end = True

        # Simulation closeout
        self.closeout_simulation()


    def closeout_eigenvalue_iteration(self):    
        # Next iteration?
        if self.i_iter < self.N_iter:
            simulation_end = False

            # Rebase RNG for population control
            mcdc.random.rng.skip_ahead(mcdc.mpi.work_size_total, change_base=True)

            # Population control stored bank
            self.bank_stored = self.popctrl(self.bank_stored, self.N_hist, 
                                            normalize=True)
            
            # Set stored bank as source bank for the next iteration
            self.bank_source = self.bank_stored
            self.bank_stored = []

        else:
            self.bank_source = []
            self.bank_stored = []
            simulation_end = True
        
        # Progress printout
        #   TODO: make optional. print in a table format
        '''
        if mcdc.mpi.master:
            print(self.i_iter,self.k_eff)
            sys.stdout.flush()
        '''
        return simulation_end 


    def closeout_simulation(self):
        # Save tallies
        if mcdc.mpi.master:
            with h5py.File(self.output+'.h5', 'w') as f:
                # Tallies
                for T in self.tallies:
                    f.create_dataset(T.name+"/energy_grid", data=T.filter_energy.grid)
                    f.create_dataset(T.name+"/angular_grid", data=T.filter_angular.grid)
                    f.create_dataset(T.name+"/time_grid", data=T.filter_time.grid)
                    f.create_dataset(T.name+"/spatial_grid", data=T.filter_spatial.grid)
                    
                    # Scores
                    for S in T.scores:
                        f.create_dataset(T.name+"/"+S.name+"/mean", data=np.squeeze(S.mean))
                        f.create_dataset(T.name+"/"+S.name+"/sdev", data=np.squeeze(S.sdev))
                        S.mean.fill(0.0)
                        S.sdev.fill(0.0)
                    
                # Eigenvalues
                if self.mode_eigenvalue:
                    f.create_dataset("keff", data=self.k_mean)
                    self.k_mean.fill(0.0)

        # Reset simulation parameters
        self.i_iter = 0


    # =========================================================================
    # SOURCE LOOP
    # =========================================================================
    
    def loop_source(self):
        # Work index
        i_work = mcdc.mpi.work_start
        while i_work < mcdc.mpi.work_end:
            # Initialize RNG wrt work index
            mcdc.random.rng.skip_ahead(i_work)

            # Get a source particle and put into history bank
            if not self.bank_source:
                # Initial source
                P = self.source.get_particle()
                # Determine cell if not given
                if not P.cell: 
                    P.cell = self.find_cell(P)
                self.bank_history.append(P)
            else:
                self.bank_history.append(self.bank_source[i_work-mcdc.mpi.work_start])
            
            # History loop
            self.loop_history()

            # Increment work index
            i_work += 1
            
            # Super rough estimate of progress
            #   TODO: Make it optional. A progress bar?
            '''
            if mcdc.mpi.master and self.mode_fixed_source:
                prog = (self.N_hist - len(self.bank_source))/self.N_hist*100
                print('%.2f'%prog,'%')
                sys.stdout.flush()
            '''
        
        # Tally source closeout
        for T in self.tallies:
            T.closeout_source(mcdc.mpi.work_size_total, self.i_iter)

        # Eigenvalue source closeout
        if self.mode_eigenvalue:
            # MPI Reduce nuSigmaF
            mcdc.mpi.allreduce(self.nuSigmaF_sum, self.nuSigmaF_buff)
            
            # Update keff
            self.k_eff = self.nuSigmaF_buff[0]/mcdc.mpi.work_size_total
            if mcdc.mpi.master:
                self.k_mean[self.i_iter] = self.k_eff
                        
            # Reset accumulator
            self.nuSigmaF_sum = 0.0


    def find_cell(self, P):
        pos = P.pos
        C = None
        for cell in self.cells:
            if cell.test_point(pos):
                C = cell
                break
        if C == None:
            print("ERROR: A particle is lost at "+str(pos))
            sys.exit()
        return C
        
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
            # Record initial parameters
            P.save_previous_state()

            # Get speed and XS (not neeeded for MG mode)
            P.speed = self.speeds[P.g]
        
            # Collision distance
            d_coll = self.get_collision_distance(P)

            # Nearest surface and distance to hit it
            S, d_surf = self.surface_distance(P)
            
            # Move particle
            d_move = min(d_coll, d_surf)
            self.move_particle(P, d_move)

            # Surface hit or collision?
            if d_coll > d_surf:
                # Record surface hit
                P.surface = S                              
                self.surface_hit(P)
            else:
                self.collision(P)
                            
            # Score tallies
            for T in self.tallies:
                T.score(P)
                
            # Score eigenvalue tallies
            if self.mode_eigenvalue:
                nu       = P.cell_old.material.nu[P.g_old]
                SigmaF   = P.cell_old.material.SigmaF_tot[P.g_old]
                nuSigmaF = nu*SigmaF
                self.nuSigmaF_sum += P.wgt_old*P.distance*nuSigmaF
            
            # Reset particle record
            P.reset_record()           
                
    # =========================================================================
    # Particle transports
    # =========================================================================

    def get_collision_distance(self, P):
        xi     = mcdc.random.rng()
        SigmaT = P.cell.material.SigmaT[P.g]
        d_coll = -np.log(xi)/SigmaT
        return d_coll

    # Get nearest surface and distance to hit for particle P
    def surface_distance(self,P):
        S     = None
        d_surf = np.inf
        for surf in P.cell.surfaces:
            d = surf[0].distance(P.pos, P.dir)
            if d < d_surf:
                S     = surf[0];
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
        self.move_particle(P, SMALL_KICK)
        
        # Get new cell
        if P.alive:
            P.cell = self.find_cell(P)
            
    def collision(self, P):
        P.collision = True
        SigmaT = P.cell.material.SigmaT[P.g]
        SigmaS = P.cell.material.SigmaS_tot[P.g]
        SigmaF = P.cell.material.SigmaF_tot[P.g]
        
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
        SigmaS     = P.cell.material.SigmaS[P.g]
        SigmaS_tot = P.cell.material.SigmaS_tot[P.g]
        G          = len(SigmaS)
        
        # Sample outgoing energy
        xi  = mcdc.random.rng()*SigmaS_tot
        tot = 0.0
        for g_out in range(G):
            tot += SigmaS[g_out][0]
            if tot > xi:
                break
        P.g = g_out
        
        # Sample scattering angle
        if len(SigmaS[g_out]) == 1:
            # Isotropic
            mu0 = 2.0*mcdc.random.rng() - 1.0;
        # TODO: sampling anisotropic scattering with rejection sampling
        
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
        SigmaF     = P.cell.material.SigmaF[P.g]
        SigmaF_tot = P.cell.material.SigmaF_tot[P.g]
        nu         = P.cell.material.nu[P.g]
        G          = len(SigmaF)
        
        # Kill the current particle
        P.alive = False
        
        # Sample number of fission neutrons
        #   in fixed-source, k_eff = 1.0
        N = floor(nu/self.k_eff + mcdc.random.rng())
        P.fission_neutrons = N

        # Push fission neutrons to bank
        for n in range(N):
            # Sample outgoing energy
            xi  = mcdc.random.rng()*SigmaF_tot
            tot = 0.0
            for g_out in range(G):
                tot += SigmaF[g_out]
                if tot > xi:
                    break
            
            # Sample isotropic direction
            dir = self.isotropic_dir.sample()
            
            # Bank
            self.bank_fission.append(Particle(P.pos, dir, g_out, P.time, P.wgt, P.cell))
