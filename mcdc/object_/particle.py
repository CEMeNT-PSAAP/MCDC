import numpy as np

from dataclasses import dataclass, field
from typing import Annotated
from numpy import int64, uint64
from numpy.typing import NDArray

####

from mcdc.constant import PARTICLE_NEUTRON
from mcdc.object_.base import MCDCBase, MCDCBase


@dataclass
class ParticleData(MCDCBase):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    t: float = 0.0
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 0.0
    g: int = -1
    E: float = 0.0
    w: float = 0.0
    particle_type: int = PARTICLE_NEUTRON
    rng_seed: uint64 = uint64(1)

    def __post_init__(self):
        # MC/DC framework metadata
        super().__init__("particle_data", [])


@dataclass
class CollisionData(MCDCBase):
    energy_deposition: float = 0.0

    def __post_init__(self):
        # MC/DC framework metadata
        super().__init__("collision_data", [])


@dataclass
class Particle(ParticleData):
    cell_ID: int = -1
    material_ID: int = -1
    surface_ID: int = -1
    alive: bool = False
    fresh: bool = False
    event: int = -1

    def __post_init__(self):
        # MC/DC framework metadata
        super(ParticleData, self).__init__("particle", [])


class ParticleBank(MCDCBase):
    particles: list[ParticleData] = []  # Non-numba
    size: Annotated[NDArray[int64], (1,)]
    tag: str = ""

    def __init__(self, tag):
        # MC/DC framework metadata
        """Initialize particle bank framework metadata."""
        super().__init__("particle_bank", ["particles"])

        self.tag = tag
        self.size = np.zeros(1, dtype=int64)
