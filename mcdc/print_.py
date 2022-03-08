import mcdc.mpi as mpi
import sys

# Get mcdc global variables/objects
import mcdc.global_ as mcdc

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
        +"  __  __  ____  __ ____   ____ \n"\
        +" |  \/  |/ ___|/ /_  _ \ / ___|\n"\
        +" | |\/| | |   /_  / | | | |    \n"\
        +" | |  | | |___ / /| |_| | |___ \n"\
        +" |_|  |_|\____|// |____/ \____|\n"
        print(banner)
        sys.stdout.flush()

def print_progress(work_idx):
    if mpi.master:
        perc = (work_idx+1.0)/mpi.work_size
        sys.stdout.write('\r')
        sys.stdout.write(" [%-28s] %d%%" % ('='*int(perc*28), perc*100.0))
        sys.stdout.flush()

def print_progress_eigenvalue(iter_idx):
    if mpi.master:
        sys.stdout.write('\r')
        sys.stdout.write("\033[K")
        if not mcdc.settings.mode_alpha:
            print(" %-4i %.5f"%(iter_idx+1,mcdc.global_tally.k_eff))
        else:
            print(" %-4i %.5f %.3e"%
                    (iter_idx+1,mcdc.global_tally.k_eff,
                     mcdc.global_tally.alpha_eff))
        sys.stdout.flush()

def print_runtime():
    if mcdc.mpi.master:
        total = mcdc.runtime_total.total
        if total >= 24*60*60:
            print(' Total runtime: %.2f days\n'%(total/24/60/60))
        elif total >= 60*60:
            print(' Total runtime: %.2f hours\n'%(total/60/60))
        elif total >= 60:
            print(' Total runtime: %.2f minutes\n'%(total/60))
        else:
            print(' Total runtime: %.2e seconds\n'%total)
        sys.stdout.flush()
