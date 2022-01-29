import sys
import mcdc.mpi

def print_error(msg):
    if mcdc.mpi.master:
        print("ERROR: %s\n"%msg)
        sys.stdout.flush()
        sys.exit()

def print_warning(msg):
    if mcdc.mpi.master:
        print("Warning: %s\n"%msg)
        sys.stdout.flush()

def print_banner():
    if mcdc.mpi.master:
        banner = "\n"\
        +"  __  __  ____  __ ____   ____ \n"\
        +" |  \/  |/ ___|/ /_  _ \ / ___|\n"\
        +" | |\/| | |   /_  / | | | |    \n"\
        +" | |  | | |___ / /| |_| | |___ \n"\
        +" |_|  |_|\____|// |____/ \____|\n"
        print(banner)
        sys.stdout.flush()
