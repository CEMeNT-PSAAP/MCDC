import numba as nb
import sys

from mpi4py import MPI
master = MPI.COMM_WORLD.Get_rank() == 0

def print_msg(msg):
    if master:
        print(msg)
        sys.stdout.flush()

def print_error(msg):
    if master:
        print("ERROR: %s\n"%msg)
        sys.stdout.flush()
        sys.exit()

def print_warning(msg):
    if master:
        print("Warning: %s\n"%msg)
        sys.stdout.flush()

def print_banner():
    size = MPI.COMM_WORLD.Get_size()
    if master:
        banner = "\n"\
        +r"  __  __  ____  __ ____   ____ "+"\n"\
        +r" |  \/  |/ ___|/ /_  _ \ / ___|"+"\n"\
        +r" | |\/| | |   /_  / | | | |    "+"\n"\
        +r" | |  | | |___ / /| |_| | |___ "+"\n"\
        +r" |_|  |_|\____|// |____/ \____|"+"\n"\
        + "\n"
        if nb.config.DISABLE_JIT:
            banner += "           Mode | Python\n"
        else:
            banner += "           Mode | Numba\n"
        banner     += "  MPI Processes | %i\n"%size
        banner     += " OpenMP Threads | 1"
        print(banner)
        sys.stdout.flush()

def print_progress(percent, mcdc):
    if master:
        sys.stdout.write('\r')
        if not mcdc['setting']['mode_eigenvalue']:
            sys.stdout.write(" [%-28s] %d%%" % ('='*int(percent*28), percent*100.0))
        else:
            if mcdc['setting']['gyration_radius']:
                sys.stdout.write(" [%-40s] %d%%" % ('='*int(percent*40), percent*100.0))
            else:
                sys.stdout.write(" [%-32s] %d%%" % ('='*int(percent*32), percent*100.0))
        sys.stdout.flush()

def print_header_eigenvalue(mcdc):
    if master:
        if mcdc['setting']['gyration_radius']:
            print("\n #     k        GyRad.  k (avg)            ")
            print(  " ====  =======  ======  ===================")
        else:
            print("\n #     k        k (avg)            ")
            print(  " ====  =======  ===================")

def print_progress_eigenvalue(mcdc):
    if master:
        i_cycle = mcdc['i_cycle']
        k_eff   = mcdc['k_eff']
        k_avg   = mcdc['k_avg_running']
        k_sdv   = mcdc['k_sdv_running']
        gr      = mcdc['gyration_radius'][i_cycle]

        sys.stdout.write('\r')
        sys.stdout.write("\033[K")
        if mcdc['setting']['gyration_radius']:
            if not mcdc['cycle_active']:
                print(" %-4i  %.5f  %6.2f"%(i_cycle+1,k_eff,gr))
            else:
                print(" %-4i  %.5f  %6.2f  %.5f +/- %.5f"%(i_cycle+1,k_eff,gr,k_avg,k_sdv))
        else:
            if not mcdc['cycle_active']:
                print(" %-4i  %.5f"%(i_cycle+1,k_eff))
            else:
                print(" %-4i  %.5f  %.5f +/- %.5f"%(i_cycle+1,k_eff,k_avg,k_sdv))
        sys.stdout.flush()

def print_runtime(mcdc):
    total = mcdc['runtime_total']
    if master:
        if total >= 24*60*60:
            print(' Total runtime: %.2f days\n'%(total/24/60/60))
        elif total >= 60*60:
            print(' Total runtime: %.2f hours\n'%(total/60/60))
        elif total >= 60:
            print(' Total runtime: %.2f minutes\n'%(total/60))
        else:
            print(' Total runtime: %.2f seconds\n'%total)
        sys.stdout.flush()

def print_bank(bank, show_content=False):
    tag       = bank['tag']
    size      = bank['size']
    max_size  = bank['max_size']
    particles = bank['particles']

    print('\n=============')
    print('Particle bank')
    print('  tag  :', tag)
    print('  size :', size, 'of', max_size)
    if show_content and size > 0:
        for i in range(size):
            print(' ',particles[i])
    print('\n')
