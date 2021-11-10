import numpy as np


# =============================================================================
# Material
# =============================================================================

class Material:
    """
    Material to fill `mcdc.Cell` object

    In multi-group mode, it is a structure of cross-sections.

    Attributes
    ----------
    SigmaT : numpy.ndarray (1D)
        Total cross-section
    SigmaC : numpy.ndarray (1D)
        Capture cross-section
    SigmaS : numpy.ndarray (1D)
        Scattering cross-section
    SigmaF : numpy.ndarray (1D)
        Fission cross-section
    nu : numpy.ndarray (1D)
        Fission multiplication
    SigmaS_diff : numpy.ndarray (2D)
        Differential scattering cross-section
    SigmaF_diff : numpy.ndarray (2D)
        Differential fission cross-section
    G : int
        Multi-group size
    """

    def __init__(self, SigmaC, SigmaS, SigmaF, nu):
        """
        Arguments
        ---------
        SigmaC : numpy.ndarray (1D)
            Capture cross-section
        nu : numpy.ndarray (1D)
            Fission multiplication
        SigmaS : numpy.ndarray (2D)
            Differential scattering cross-section [out][in]
        SigmaF : numpy.ndarray (2D)
            Differential fission cross-section [out][in]
        """

        self.SigmaC = SigmaC
        self.nu     = nu
        self.G      = len(nu)

        # Scattering and fission
        self.SigmaS = np.sum(SigmaS,0)
        self.SigmaF = np.sum(SigmaF,0)
        self.SigmaS_diff = np.swapaxes(SigmaS,0,1) # [out,in] -> [in,out]    
        self.SigmaF_diff = np.swapaxes(SigmaF,0,1) # [out,in] -> [in,out]
        
        # Total
        self.SigmaT = self.SigmaC + self.SigmaS + self.SigmaF
