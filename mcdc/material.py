import numpy as np


# =============================================================================
# Material
# =============================================================================

class Material:
    def __init__(self, SigmaT, SigmaS, nu, SigmaF):
        self.SigmaT = SigmaT
        self.nu     = nu

        # If only isotropic SigmaS0 is given
        if SigmaS.ndim == 2: SigmaS = np.expand_dims(SigmaS, axis=0)

        # Total scattering and fission
        self.SigmaS_tot = np.sum(SigmaS[0],0)
        self.SigmaF_tot = np.sum(SigmaF,0)
        
        # Capture
        self.SigmaC = self.SigmaT - self.SigmaS_tot - self.SigmaF_tot
            
        # Matrices
        self.SigmaS = np.swapaxes(SigmaS,0,2) # [ord,out,in] -> [in,out,ord]    
        self.SigmaF = np.swapaxes(SigmaF,0,1) # [out,in]     -> [in,out]