import sys

import numpy as np
import matplotlib.pyplot as plt

from mcdc.print import print_error, print_warning


# =============================================================================
# Material
# =============================================================================

class Material:
    """
    Material to fill mcdc.Cell object

    In multi-group mode, it is a structure of cross-sections.

    Attributes
    ----------
    G : int
        # of energy groups
    J : int
        # of delayed neutron precursor groups

    speed : numpy.ndarray (1D)
        Energy group speed
    decay : numpy.ndarray (1D)
        Precursor group decay constant [/s]

    total : numpy.ndarray (1D)
        Total cross-section [/cm]
    capture : numpy.ndarray (1D)
        Capture cross-section [/cm]
    scatter : numpy.ndarray (1D)
        Scattering cross-section [/cm]
    fission : numpy.ndarray (1D)
        Fission cross-section [/cm]

    nu_p : numpy.ndarray (1D)
        Prompt fission neutron yield
    nu_d : numpy.ndarray (2D)
        Delayed neutron precursor yield [in][dgroup]

    chi_s : numpy.ndarray (2D)
        Scattering spectrum [in][out]
    chi_p : numpy.ndarray (2D)
        Prompt fission spectrum [in][out]
    chi_d : numpy.ndarray (2D)
        Delayed neutron spectrum [dgroup][out]
    """

    def __init__(self, capture=None, scatter=None, fission=None, nu_p=None, 
                 nu_d=None, chi_p=None, chi_d=None, speed=None, decay=None):
        """
        Arguments
        ---------
        capture : numpy.ndarray (1D)
            Capture cross-section [/cm]
        scatter : numpy.ndarray (2D)
            Differential scattering cross-section [gout][gin] [/cm].
        fission : numpy.ndarray (1D)
            Fission cross-section [/cm]. 
        *At least capture, scatter, or fission cross-section needs to be 
        provided.

        nu_p : numpy.ndarray (1D)
            Prompt fission neutron yield.
        nu_d : numpy.ndarray (2D)
            Delayed neutron precursor yield [dg][gin].
        *nu_p or nu_d is needed if fission is provided.
        
        chi_p : numpy.ndarray (2D)
            Prompt fission spectrum [gout][gin]
        chi_d : numpy.ndarray (2D)
            Delayed neutron spectrum [gout][dg]
        *chi_p and chi_d are needed if nu_p and nu_d are provided, respectively.

        speed : numpy.ndarray (1D)
            Energy group speed
        decay : numpy.ndarray (1D)
            Precursor group decay constant [/s]
        *speed and decay are optional. By default, values for speed and decay 
        are one and infinite, respectively. Universal speed and decay can be 
        provided through mcdc.simulator.
        """

        # Energy group size
        if capture is not None:
            self.G = len(capture)
        elif scatter is not None:
            self.G = len(scatter)
        elif fission is not None:
            self.G = len(fission)
        else:
            print_error("Need to supply capture, scatter, or fission to "\
                        + "mcdc.Material")

        # Delayed group size
        if nu_d is not None:
            self.J = len(nu_d)
        else:
            self.J = 1
        
        # Cross-sections
        if capture is not None:
            self.capture = capture
        else:
            self.capture = np.zeros(self.G)
        if scatter is not None:
            self.scatter = np.sum(scatter,0)
        else:
            self.scatter = np.zeros(self.G)
        if fission is not None:
            if fission.ndim == 1:
                self.fission = fission
            else:
                self.fission = np.sum(fission,0)
        else:
            self.fission = np.zeros(self.G)
        
        # Fission productions
        if fission is not None:
            if nu_p is None and nu_d is None:
                print_error("Need to supply nu_p or nu_d for fissionable "\
                            + "mcdc.Material")
        if nu_p is not None:
            self.nu_p = nu_p
        else:
            self.nu_p = np.zeros(self.G)
        if nu_d is not None:
            self.nu_d  = np.copy(nu_d)
            self.nu_d  = np.swapaxes(self.nu_d,0,1) # [dg,gin] -> [gin,dg]
        else:
            self.nu_d = np.zeros([self.G,1])

        # Scattering spectrum
        if scatter is not None:
            self.chi_s = np.copy(scatter)
            self.chi_s = np.swapaxes(self.chi_s,0,1) # [gout,gin] -> [gin,gout]
            for g in range(self.G): 
                if self.scatter[g] > 0.0:
                    self.chi_s[g,:] /= self.scatter[g]
        else:
            self.chi_s = np.zeros([self.G,self.G])

        # Fission spectrums
        if nu_p is not None:
            if self.G == 1:
                self.chi_p = np.array([[1.0]])
            elif chi_p is None:
                print_error("Need to supply chi_p if nu_p is provided")
            else:
                self.chi_p = np.copy(chi_p)
                self.chi_p = np.swapaxes(self.chi_p,0,1) # [gout,gin] -> [gin,gout]
                # Normalize
                for g in range(self.G):
                    if np.sum(chi_p[g,:]) > 0.0:
                        chi_p[g,:] /= np.sum(chi_p[g,:])
        if nu_d is not None:
            if chi_d is None:
                print_error("Need to supply chi_d if nu_d is provided")
            self.chi_d = np.copy(chi_d)
            self.chi_d = np.swapaxes(self.chi_d,0,1) # [gout,dg] -> [dg,gout]
            # Normalize
            for dg in range(self.J):
                if np.sum(chi_d[dg,:]) > 0.0:
                    chi_d[dg,:] /= np.sum(chi_d[dg,:])

        # Get speed
        if speed is not None:
            self.speed = speed
        else:
            self.speed = np.ones(self.G)

        # Get decay constant
        if decay is not None:
            self.decay = decay
        else:
            self.decay = np.ones(self.J)*np.inf
 
        # Total
        self.total = self.capture + self.scatter + self.fission
