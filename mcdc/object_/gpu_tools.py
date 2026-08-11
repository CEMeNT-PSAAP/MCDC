from dataclasses import dataclass
from numpy import uintp

####

from mcdc.object_.base import MCDCBase


@dataclass
class GPUMeta(MCDCBase):
    """Opaque device pointers owned by the GPU execution bridge."""

    # MC/DC framework metadata
    label = "gpu_meta"

    state_pointer: uintp = uintp(0)
    program_pointer: uintp = uintp(0)
    simulation_pointer: uintp = uintp(0)
    data_pointer: uintp = uintp(0)

    # Note that the uintp is manually overriden in code_factory.
