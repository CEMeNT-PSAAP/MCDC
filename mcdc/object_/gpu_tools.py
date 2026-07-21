from dataclasses import dataclass
from numpy import uintp

####

from mcdc.object_.base import MCDCBase


@dataclass
class GPUMeta(MCDCBase):
    state_pointer: uintp = uintp(0)
    program_pointer: uintp = uintp(0)
    simulation_pointer: uintp = uintp(0)
    data_pointer: uintp = uintp(0)

    # Note that the uintp is manually overriden in code_factory.

    def __init__(self):
        # MC/DC framework metadata
        super().__init__("gpu_meta", [])
