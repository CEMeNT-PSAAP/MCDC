import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Material
# =============================================================================

class Material:
    """
    Material to fill `mcdc.Cell` object

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

    SigmaT : numpy.ndarray (1D)
        Total cross-section [/cm]
    SigmaC : numpy.ndarray (1D)
        Capture cross-section [/cm]
    SigmaS : numpy.ndarray (1D)
        Scattering cross-section [/cm]
    SigmaF : numpy.ndarray (1D)
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

    def __init__(self, SigmaC, SigmaS, SigmaF, nu_p, 
                       nu_d=[], chi_p=[], chi_d=[], speed=[], decay=[]):
        """
        Arguments
        ---------
        SigmaC : numpy.ndarray (1D)
            Capture cross-section
        SigmaS : numpy.ndarray (2D)
            Differential scattering cross-section [out][in]
        SigmaF : numpy.ndarray (2D) or numpy.ndarray (1D)
            Differential fission cross-section [out][in]
            or
            Fission cross-section
        nu_p : numpy.ndarray (1D)
            Prompt fission neutron yield
        nu_d : numpy.ndarray (2D)
            Delayed neutron precursor yield [dgroup][in]

        chi_p : numpy.ndarray (2D)
            Prompt fission spectrum [out][on]
        chi_d : numpy.ndarray (2D)
            Delayed neutron spectrum [out][dgroup]
    
        speed : numpy.ndarray (1D)
            Energy group speed
        decay : numpy.ndarray (1D)
            Precursor group decay constant [/s]
        """
        
        # Speed and decay constant if given
        self.speed = speed
        self.decay = decay

        # Fission productions and group sizes
        self.nu_p = nu_p
        self.G    = len(nu_p)
        if len(nu_d) > 0: 
            self.nu_d  = np.copy(nu_d)
            self.nu_d  = np.swapaxes(self.nu_d,0,1) # [dgroup,in] -> [in,dgroup]
            self.J = len(nu_d[0])
        else:
            self.nu_d  = nu_d
            self.J = 0

        # Cross-sections
        self.SigmaC = SigmaC
        if SigmaF.ndim == 1:
            self.SigmaF = SigmaF
        else:
            self.SigmaF = np.sum(SigmaF,0)
        self.SigmaS = np.sum(SigmaS,0)

        # Spectrums
        self.chi_s = np.copy(SigmaS)
        self.chi_s = np.swapaxes(self.chi_s,0,1) # [out,in] -> [in,out]
        for g in range(self.G): 
            if self.SigmaS[g] > 0.0:
                self.chi_s[g,:] = self.chi_s[g,:]/self.SigmaS[g]
        if SigmaF.ndim == 2:
            self.chi_p = np.copy(SigmaF)
            self.chi_p = np.swapaxes(self.chi_p,0,1) # [out,in] -> [in,out]
            for g in range(self.G): 
                if self.SigmaF[g] > 0.0:
                    self.chi_p[g,:] = self.chi_p[g,:]/self.SigmaF[g]
        else:
            self.chi_p = np.copy(chi_p)
            self.chi_p = np.swapaxes(self.chi_p,0,1) # [out,in] -> [in,out]
        if self.J > 0: 
            self.chi_d  = np.copy(chi_d)
            self.chi_d  = np.swapaxes(self.chi_d,0,1) # [out,dgroup] -> [dgroup,out]
        else:
            self.chi_d  = chi_d
        
        # Total
        self.SigmaT = self.SigmaC + self.SigmaS + self.SigmaF
