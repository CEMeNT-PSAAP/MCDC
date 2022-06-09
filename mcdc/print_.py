import mcdc.mpi as mpi
import sys

def print_msg(msg):
    if mpi.master:
        print(msg)
        sys.stdout.flush()

def print_error(msg):
    if mpi.master:
        print("ERROR: %s\n"%msg)
        sys.stdout.flush()
        sys.exit()

def print_warning(msg):
    if mpi.master:
        print("Warning: %s\n"%msg)
        sys.stdout.flush()

def print_banner():
    if mpi.master:
        banner = "\n"\
        +r"  __  __  ____  __ ____   ____ "+"\n"\
        +r" |  \/  |/ ___|/ /_  _ \ / ___|"+"\n"\
        +r" | |\/| | |   /_  / | | | |    "+"\n"\
        +r" | |  | | |___ / /| |_| | |___ "+"\n"\
        +r" |_|  |_|\____|// |____/ \____|"+"\n"
        print(banner)
        sys.stdout.flush()

def print_progress(work_idx):
    if mpi.master:
        perc = (work_idx+1.0)/mpi.work_size
        sys.stdout.write('\r')
        sys.stdout.write(" [%-28s] %d%%" % ('='*int(perc*28), perc*100.0))
        sys.stdout.flush()

def print_progress_eigenvalue(mcdc):
    if mpi.master:
        i_iter = mcdc['i_iter']
        k_eff = mcdc['k_eff']
        alpha_eff = mcdc['alpha_eff']

        sys.stdout.write('\r')
        sys.stdout.write("\033[K")
        if not mcdc['setting']['mode_alpha']:
            print(" %-4i %.5f"%(i_iter+1,k_eff))
        else:
            print(" %-4i %.5f %.3e"%(i_iter+1,k_eff,alpha_eff))
        sys.stdout.flush()

def print_runtime(mcdc):
    total = mcdc['runtime_total']
    if mpi.master:
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
