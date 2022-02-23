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
                              EVENT_COLLISION, EVENT_SURFACE, EVENT_CENSUS,\
                              EVENT_MESH
from mcdc.random       import RandomLCG
from mcdc.pct          import *
from mcdc.misc         import binary_search
from mcdc.print        import print_banner, print_error
from mcdc.vrt          import WeightWindow


class Simulator:
    def __init__(self, cells=[], sources=[], N_hist = 0, speed=[], decay=[]):

        # Model
        self.cells   = cells   # list of mcdc.Cell
        self.sources = sources # list of mcdc.Source
        self.speed   = speed   # array of particle MG speeds (for univerasl use)
        self.decay   = decay   # array of precursor group decay constants
                               # (for universal use)
        self.N_hist  = int(N_hist) # number of histories (per iteration)

        # Output
        self.output = "output" # .h5 output file name
        self.tally  = None
        
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

        # Variance reduction settings
        self.implicit_capture = False
        self.weight_window = None

        # Particle banks
        #   TODO: use fixed memory allocations with helper indices
        self.bank_stored  = [] # for the next source loop
        self.bank_source  = [] # for current source loop
        self.bank_history = [] # for current history loop
        self.bank_fission = None # will point to one of the banks

        # Timer
        self.time_total = 0.0

        # Misc.
        self.G             = len(self.speed) # Number of energy groups
        self.J             = len(self.speed) # Number of delayed groups
        self.isotropic_dir = DistPointIsotropic() # (see distribution.py)
        self.parallel_hdf5 = False # TODO
            
    def set_tally(self, scores, x=None, y=None, z=None, t=None):
        self.tally = mcdc.Tally('tally', scores, x, y, z, t)

    # =========================================================================
    # Feature setters
    # =========================================================================
    
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
            print_error("Unknown PCT "+pct)

        # Set census time
        self.census_time = census_time

    def set_weight_window(self, x=None, y=None, z=None, t=None, window=None):
        self.weight_window = WeightWindow(x,y,z,t,window)

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
        
        # Get energy and delayed groups
        self.G = self.cells[0].material.G
        self.J = self.cells[0].material.J

        # Set group universal speed and decay constants if given
        if len(self.speed) > 0:
            for c in self.cells:
                c.material.speed = self.speed
        if len(self.decay) > 0:
            for c in self.cells:
                c.material.decay = self.decay

        # Allocate tally bins
        if self.tally is None:
            self.tally = mcdc.Tally('tally', ['flux'])
        self.tally.allocate_bins(self.N_iter, self.G)

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
                self.tally.closeout(self.N_hist, self.i_iter)

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
            self.tally.closeout(self.N_hist, 0)

        # Stop timer
        self.time_total = mcdc.mpi.Wtime() - self.time_total

        # Save tallies to HDF5
        if mcdc.mpi.master and self.tally is not None:
            with h5py.File(self.output+'.h5', 'w') as f:
                # Runtime
                f.create_dataset("runtime",data=np.array([self.time_total]))
                f.create_dataset("runtime_pct",data=np.array([self.time_pct]))

                # Tally
                T = self.tally
                f.create_dataset(T.name+"/grid/t", data=T.t)
                f.create_dataset(T.name+"/grid/x", data=T.x)
                f.create_dataset(T.name+"/grid/y", data=T.y)
                f.create_dataset(T.name+"/grid/z", data=T.z)
                
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
                if P.cell is None:
                    self.set_cell(P)
                if P.idx_census_time is None:
                    self.set_census_time_idx(P)

                self.bank_history.append(P)
            else:
                self.bank_history.append(self.bank_source[i])
            
            if self.weight_window is not None:
                self.weight_window(P, self.bank_history)

            # History loop
            self.loop_history()
            
            # Progress printout
            if mcdc.mpi.master:
                perc = (i+1.0)/mcdc.mpi.work_size
                sys.stdout.write('\r')
                sys.stdout.write(" [%-28s] %d%%" % ('='*int(perc*28), perc*100.0))
                sys.stdout.flush()

    
    def set_cell(self, P):
        position = P.position
        time = P.time
        C = None
        for cell in self.cells:
            if cell.test_point(position,time):
                C = cell
                break
        if C == None:
            print_error("A particle is lost at "+str(position))
            sys.exit()
        P.cell = C
        
    def set_census_time_idx(self, P):
        t = P.time
        idx = binary_search(t, self.census_time) + 1

        if idx == len(self.census_time):
            P.alive = False
            idx = None
        elif P.time == self.census_time[idx]:
            idx += 1
        P.idx_census_time = idx


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
        self.tally.closeout_history()
            
    # =========================================================================
    # PARTICLE LOOP
    # =========================================================================
    
    def loop_particle(self, P):
        while P.alive:
            P.speed = P.cell.material.speed[P.group]

            # =================================================================
            # Get distances to events
            # =================================================================
    
            # Distance to collision
            d_coll = self.collision_distance(P)

            # Nearest surface and distance to hit
            S, d_surf = self.surface_distance(P)

            # Distance to mesh
            d_mesh = self.tally.distance(P)

            # Distance to census
            t_census = self.census_time[P.idx_census_time]
            d_census = P.speed*(t_census - P.time)

            # =================================================================
            # Get event
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
            if d_move > d_mesh:
                event  = EVENT_MESH
                d_move = d_mesh
            
            # =================================================================
            # Score track-length tallies
            # =================================================================

            self.tally.score(P, d_move)
            
            # Score eigenvalue tallies
            if self.mode_eigenvalue:
                nu       = P.cell.material.nu_p[P.group]\
                           + sum(P.cell.material.nu_d[P.group])
                SigmaF   = P.cell.material.fission[P.group]
                nuSigmaF = nu*SigmaF
                self.nuSigmaF_sum += P.weight*d_move*nuSigmaF

                if self.mode_alpha:
                    self.ispeed_sum += P.weight*d_move\
                                       /P.cell.material.speed[P.group]

            # =================================================================
            # Move to event
            # =================================================================
    
            # Move particle
            self.move_particle(P, d_move)
            
            # =================================================================
            # Perform event
            # =================================================================    

            if event == EVENT_SURFACE:
                # Record surface hit
                P.surface = S

                # Implement surface hit
                self.surface_hit(P)

            elif event == EVENT_COLLISION:
                # Sample collision
                self.collision(P)
 
            elif event == EVENT_CENSUS:
                # Cross the time boundary
                d = SMALL*P.speed
                self.move_particle(P, d)

                # Increment index
                P.idx_census_time += 1
                # Not final census?
                if P.idx_census_time < len(self.census_time):
                    # Store for next time census
                    self.bank_stored.append(P.create_copy())
                P.alive = False

            elif event == EVENT_MESH:
                # Small kick (see constant.py) to make sure crossing
                self.move_particle(P, SMALL)
        
            
            # =================================================================
            # Weight window
            # =================================================================    
            
            if P.alive and self.weight_window is not None:
                self.weight_window(P, self.bank_history)

    # =========================================================================
    # Particle transports
    # =========================================================================

    def collision_distance(self, P):
        xi     = mcdc.random.rng()
        SigmaT = P.cell.material.total[P.group]

        SigmaT += VERY_SMALL # To ensure non-zero value

        if self.mode_alpha:
            SigmaT += abs(self.alpha_eff)/P.speed

        d_coll  = -np.log(xi)/SigmaT
        return d_coll

    # Get nearest surface and distance to hit for particle P
    def surface_distance(self,P):
        S     = None
        d_surf = np.inf
        for surf in P.cell.surfaces:
            speed = P.cell.material.speed[P.group]
            d = surf.distance(P.position, P.direction, P.time, speed)
            if d < d_surf:
                S     = surf;
                d_surf = d;
        return S, d_surf

    def move_particle(self, P, d):
        # 4D Move
        P.position += P.direction*d
        P.time     += d/P.speed
        
    def surface_hit(self, P):
        # Implement BC
        P.surface.bc(P)

        # Small kick (see constant.py) to make sure crossing
        self.move_particle(P, SMALL)
        
        # Set new cell
        if P.alive:
            self.set_cell(P)
            
    def collision(self, P):
        SigmaT = P.cell.material.total[P.group]
        SigmaC = P.cell.material.capture[P.group]
        SigmaS = P.cell.material.scatter[P.group]
        SigmaF = P.cell.material.fission[P.group]

        if self.mode_alpha:
            Sigma_alpha = abs(self.alpha_eff)/P.cell.material.speed[P.group]
            SigmaT += Sigma_alpha

        if self.implicit_capture:
            capture = SigmaC
            if self.mode_alpha:
                capture += Sigma_alpha
            P.weight *= (SigmaT-capture)/SigmaT
            SigmaT -= capture

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
        chi_s = P.cell.material.chi_s[P.group]
        G     = P.cell.material.G
        
        # Sample outgoing energy
        xi  = mcdc.random.rng()
        tot = 0.0
        for g_out in range(G):
            tot += chi_s[g_out]
            if tot > xi:
                break
        P.group = g_out
        
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

        ux = P.direction.x
        uy = P.direction.y
        uz = P.direction.z
        
        dir_final = Point(0,0,0)
    
        if uz != 1.0:
            B = (1.0 - P.direction.z**2)**0.5
            C = Ac/B
            
            dir_final.x = ux*mu + (ux*uz*cos_azi - uy*sin_azi)*C
            dir_final.y = uy*mu + (uy*uz*cos_azi + ux*sin_azi)*C
            dir_final.z = uz*mu - cos_azi*Ac*B
        
        # If dir = 0i + 0j + k, interchange z and y in the scattering formula
        else:
            B = (1.0 - uy**2)**0.5
            C = Ac/B
            
            dir_final.x = ux*mu + (ux*uy*cos_azi - uz*sin_azi)*C
            dir_final.z = uz*mu + (uz*uy*cos_azi + ux*sin_azi)*C
            dir_final.y = uy*mu - cos_azi*Ac*B
            
        P.direction = dir_final
        
    def collision_fission(self, P):
        # Kill the current particle
        P.alive = False

        # Get group numbers
        G = P.cell.material.G
        J = P.cell.material.J

        # Total nu
        nu_p = P.cell.material.nu_p[P.group]
        nu = nu_p
        if J>0: 
            nu_d = P.cell.material.nu_d[P.group]
            nu += sum(nu_d)

        # Sample number of fission neutrons
        #   in fixed-source, k_eff = 1.0
        N = floor(P.weight*nu/self.k_eff + mcdc.random.rng())

        # Push fission neutrons to bank
        for n in range(N):
            # Determine if it's prompt or delayed neutrons, 
            # then get the energy spectrum and decay constant
            xi  = mcdc.random.rng()*nu
            tot = nu_p
            # Prompt?
            if tot > xi:
                spectrum = P.cell.material.chi_p[P.group]
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
            direction = self.isotropic_dir.sample()
           
            # Create the outgoing particle
            P_out = Particle(P.position, direction, g_out, t_out, 1.0)
            P_out.cell = P.cell
            self.set_census_time_idx(P_out)

            # Bank
            self.bank_fission.append(P_out)
