import numpy as np
import sys
from math import floor
from mpi4py import MPI
import h5py
import copy

from particle import Point, Particle
from distribution import DistPointIsotropic
from constant import SMALL_KICK
import rng


class Simulator:
    def __init__(self, speeds, cells, source, N_hist = 1, tallies = [],
                 k_mode = False, k_init = 1.0, N_iter = 1,
                 output = 'output', population_control='simple',
                 seed = 777):

        # Basic settings
        self.speeds  = speeds  # array of particle MG speeds
        self.cells   = cells   # list of Cells (see geometry.py)
        self.source  = source  # Source (see particle.py)
        self.N_hist  = N_hist  # number of histories
        self.tallies = tallies # list of Tallies (see tally.py)
        self.output  = output  # .h5 output file name
        
        # Time-dependent mode settings
        #   TODO: Census and outer census
        self.popctrl = population_control # type of popctrl

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
        self.bank_survive = [] # for populaton control
        self.bank_source  = [] # for current source loop
        self.bank_history = [] # for current history loop
        self.bank_fission = None # will point to one of the banks
        
        # MPI parameters
        comm = MPI.COMM_WORLD
        self.MPI_size  = comm.Get_size()
        self.MPI_rank  = comm.Get_rank()
        self.MPI_left  = self.MPI_rank - 1
        self.MPI_right = self.MPI_rank + 1
        # History size and global indices (determined in self.run)
        self.MPI_hist_size  = 0
        self.MPI_hist_start = 0
        self.MPI_hist_end   = 0

        # RNG settings
        #   TODO: use RNG that allows stride for more flexible simulation
        self.seed        = seed
        self.rng_history = []   # rng for each history
        self.rng_popctrl = None # rng for populaton control
        
        # Variance Reduction Technique settings
        #   TODO: implicit capture, implicit fission production
        
        # Misc.
        self.isotropic_dir = DistPointIsotropic() # (see distribution.py)
            
        # Eigenvalue-specific tallies
        if self.mode_eigenvalue:
            if self.MPI_rank == 0:
                self.k_mean = np.zeros(self.N_iter) # iteration solutions for k
            # Accumulators
            self.nuSigmaF_sum = 0.0
            # MPI buffer
            self.nuSigmaF_recv = np.array([0.0]) # For MPI reduce
            
        # Set tally bins
        for T in tallies:
            T.setup_bins(N_iter) # Allocate tally bins (see tally.py)
    
    # =========================================================================
    # Run ("main")
    # =========================================================================
    
    def run(self):
        # Prepare RNG child streams
        #   child streams       --> eigenvalue iterations
        #   grand child streams --> histories
        ss = np.random.SeedSequence(self.seed)
        if self.mode_eigenvalue:
            # Reserve an additional branch (the last one) for population control
            self.seed_child  = ss.spawn(self.N_iter+1)
            
            # Set RNG for population control
            seed             = self.seed_child[-1]
            self.rng_popctrl = np.random.default_rng(seed).uniform
        else:
            self.seed_child = ss
            
        # Initialize source bank (also initialize corresponding history RNGs)
        self.set_initial_source()
            
        # Start iteration
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
            
            # Any stored bank?
            if self.bank_stored:
                self.manage_stored_bank() # also initialize corresponding RNGs
            
            # Eigenvalue iteration closeout
            if self.mode_eigenvalue: 
                self.closeout_eigenvalue_iteration()

            # End simulation?
            if self.mode_eigenvalue and self.i_iter == self.N_iter:
                simulation_end = True
            elif self.mode_fixed_source:
                simulation_end = True

        # Simulation closeout
        self.closeout_simulation()

    # =========================================================================
    # Set initial source (or initial guess for eigenvalue mode)
    #   Particle sources are evenly distributed to all MPI processors
    # =========================================================================
    
    def set_initial_source(self):
        # Calculate # of work (source particles or histories) per processor
        self.MPI_hist_size = floor(self.N_hist/self.MPI_size)
        if self.MPI_rank < self.N_hist%self.MPI_size:
            self.MPI_hist_size += 1

        # History global indices
        buff = np.array([0],dtype=int)
        MPI.COMM_WORLD.Exscan(np.array(self.MPI_hist_size,dtype=int), buff, MPI.SUM)
        self.MPI_hist_start = buff[0]
        self.MPI_hist_end   = buff[0] + self.MPI_hist_size - 1
        
        # Initialize history RNGs
        self.initialize_rng(self.MPI_hist_start, self.MPI_hist_end+1, self.N_hist)
                        
        # Get initial source particles
        for i in range(self.MPI_hist_size):
            rng.uniform = self.rng_history[i]
            P = self.source.get_particle()
            # Determine cell if not given
            if not P.cell: 
                P.cell = self.find_cell(P)
            self.bank_source.append(P)

    # =========================================================================
    # Initialize history RNGs
    # =========================================================================

    def initialize_rng(self, start, end, N):
        if self.mode_eigenvalue:
            seed_grand_child = self.seed_child[self.i_iter].spawn(N)
        else:
            seed_grand_child = self.seed_child.spawn(N)
        
        self.rng_history = [np.random.default_rng(seed).uniform 
                            for seed in seed_grand_child[start:end]]
            
    # =========================================================================
    # SOURCE LOOP
    # =========================================================================
    
    def loop_source(self):
        N_init = len(self.bank_source)
        while self.bank_source:
            # Get a particle from source bank and put into history bank
            self.bank_history.append(self.bank_source.pop())
            
            # Get the corresponding RNG as well
            rng.uniform = self.rng_history.pop()
            
            # History loop
            self.loop_history()
            
            # Super rough estimate of progress
            #   TODO: A progress bar would be nice?
            if self.MPI_rank == 0 and self.mode_fixed_source:
                prog = (N_init - len(self.bank_source))/N_init*100
                print('%.2f'%prog,'%')
                sys.stdout.flush()
        
        # Tally source closeout
        for T in self.tallies:
            T.closeout_source(self.N_hist, self.i_iter)

        # Eigenvalue source closeout
        if self.mode_eigenvalue:
            # MPI Reduce
            comm = MPI.COMM_WORLD
            comm.Allreduce(self.nuSigmaF_sum, self.nuSigmaF_recv, MPI.SUM)
            
            # Update keff
            self.k_eff = self.nuSigmaF_recv[0]/self.N_hist
            if self.MPI_rank == 0:
                self.k_mean[self.i_iter] = self.k_eff
                        
            # Reset accumulator
            self.nuSigmaF_sum = 0.0

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
        
    def get_collision_distance(self, P):
        xi     = rng.uniform()
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
        P.distance = d
        
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
        xi = rng.uniform()*SigmaT
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
        xi  = rng.uniform()*SigmaS_tot
        tot = 0.0
        for g_out in range(G):
            tot += SigmaS[g_out][0]
            if tot > xi:
                break
        P.g = g_out
        
        # Sample scattering angle
        if len(SigmaS[g_out]) == 1:
            # Isotropic
            mu0 = 2.0*rng.uniform() - 1.0;
        # TODO: sampling anisotropic scattering with rejection sampling
        
        # Scatter particle
        self.scatter(P,mu0)

    # Scatter direction with scattering cosine mu
    def scatter(self, P, mu):
        # Sample azimuthal direction
        azi     = 2.0*np.pi*rng.uniform()
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
        N = floor(nu/self.k_eff + rng.uniform())
        P.fission_neutrons = N

        # Push fission neutrons to bank
        for n in range(N):
            # Sample outgoing energy
            xi  = rng.uniform()*SigmaF_tot
            tot = 0.0
            for g_out in range(G):
                tot += SigmaF[g_out]
                if tot > xi:
                    break
            
            # Sample isotropic direction
            dir = self.isotropic_dir.sample()
            
            # Bank
            self.bank_fission.append(Particle(P.pos, dir, g_out, P.time, P.wgt, P.cell))

    # =========================================================================
    # Manage stored bank
    # =========================================================================
    
    def manage_stored_bank(self):
        if self.popctrl == 'simple':
            self.popctrl_simple()
        elif self.popctrl == 'split-roulette':
            self.popctrl_split_roulette()
        elif self.popctrl == 'combing':
            self.popctrl_combing()
            
    # =========================================================================
    # Population control: Simple
    # =========================================================================
    #  Exactly yields N_hist particles
    #  but only applicable for uniform weight population 
    #  (e.g., analog simulations 
    #         and eigenvalue simulations with implicit fission production)
    #  TODO: MPI implementation
    
    def popctrl_simple(self):           
        # Sample N_hist surviving particles from stored bank, 
        #  and store them in surviving bank
        N_source = len(self.bank_stored)
        for i in range(self.N_hist):
            idx = floor(self.rng_popctrl()*N_source)
            self.bank_survive.append(copy.deepcopy(self.bank_stored[idx]))

        # Switch stored bank and the surviving bank
        self.bank_stored  = self.bank_survive
        self.bank_survive = []
                        
        # Normalize weight to N_hist if eigenvalue
        if self.mode_eigenvalue and self.bank_stored[0].wgt != 1.0:
            for P in self.bank_stored:
                P.wgt = 1.0
        
        # Initialize rng
        if self.i_iter < self.N_iter:
            hist_start = 0
            hist_end   = self.N_hist
            self.initialize_rng(hist_start, hist_end, self.N_hist)
    
    # =========================================================================
    # Population control: Splitting & roulette 
    # =========================================================================
    # On average yields N_hist particles
    # TODO: MPI implementation
    
    def popctrl_split_roulette(self):
        # Total weight of initial population
        w_total  = 0.0
        for P in self.bank_stored:
            w_total += P.wgt
            
        # Individual weight for surviving population
        w_survive = w_total/self.N_hist
        
        # Split & roulette each particle in stored bank
        #  put surviving particles into surviving bank
        N_source = len(self.bank_stored)
        for i in range(N_source):
            P     = self.bank_stored[i]
            
            # Surviving probability
            prob  = P.wgt/w_survive
            P.wgt = w_survive
            
            # Splitting
            n_survive = floor(prob)
            for j in range(n_survive):
                self.bank_survive.append(copy.deepcopy(P))
            
            # Russian roulette
            prob -= n_survive
            xi = self.rng_popctrl()
            if xi < prob:
                self.bank_survive.append(copy.deepcopy(P))

        # Switch stored bank and the surviving bank
        self.bank_stored  = self.bank_survive
        self.bank_survive = []
                        
        # Normalize weight to N_hist if eigenvalue
        N_source     = len(self.bank_stored)
        w_normalized = self.N_hist/N_source
        if self.mode_eigenvalue:
            for P in self.bank_stored:
                P.wgt = w_normalized
        
        # Initialize rng
        if self.i_iter < self.N_iter:
            hist_start = 0
            hist_end   = N_source
            self.initialize_rng(hist_start, hist_end, N_source)

    # =========================================================================
    # Population control: Combing
    # =========================================================================
    # Exactly yields N_hist particles
    
    def popctrl_combing(self):
        pass
    
    # =========================================================================
    # Closeouts
    # =========================================================================

    def closeout_eigenvalue_iteration(self):
        # Set stored bank as source bank for the next iteration?
        if self.i_iter < self.N_iter:
            self.bank_source = self.bank_stored
            self.bank_stored = []
            
        # Progress printout
        #   TODO: print in a table format
        if self.MPI_rank == 0:
            print(self.i_iter,self.k_eff)
            sys.stdout.flush()

    def closeout_simulation(self):
        # Save tallies
        if self.MPI_rank == 0:
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