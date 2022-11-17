import matplotlib.pyplot as plt
import numpy             as np

def plot_convergence(name, N_particle, error):
    mid   = int(len(N_particle)/2)
    line  = 1.0/np.sqrt(N_particle)
    line *= error[mid]/line[mid]
    plt.plot(N_particle, error, 'bo', fillstyle='none')
    plt.plot(N_particle, line, 'r--', label=r'$O(N^{-0.5})$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('2-norm of relative error')
    plt.xlabel(r'# of histories, $N$')
    plt.legend()
    plt.grid()
    plt.title(name)
    plt.savefig('../../'+name+'.png')
    plt.clf()
