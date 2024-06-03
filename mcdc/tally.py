# Get input_deck
import mcdc.global_ as global_

import numpy as np

from mcdc.constant import (
    INF,
    PI,
)
import mcdc.type_ as type_


def mesh_tally(
    scores,
    x=np.array([-INF, INF]),
    y=np.array([-INF, INF]),
    z=np.array([-INF, INF]),
    t=np.array([-INF, INF]),
    mu=np.array([-1.0, 1.0]),
    azi=np.array([-PI, PI]),
    g=np.array([-INF, INF]),
    E=np.array([0.0, INF]),
):
    """
    Create a tally card to collect MC solutions.

    Parameters
    ----------
    scores : list of str
        List of tally types (default ["tracklength"]).
    x : array_like[float], optional
        x-coordinates that demarcate tally bins (default numpy.ndarray([-INF, INF])).
    y : array_like[float], optional
        y-coordinates that demarcate tally bins (default numpy.ndarray([-INF, INF])).
    z : array_like[float], optional
        z-coordinates that demarcate tally bins (default numpy.ndarray([-INF, INF])).
    t : array_like[float], optional
        Times that demarcate tally bins (default numpy.ndarray([-INF, INF])).
    mu : array_like[float], optional
        Angles that demarcate axial angular tally bins (default numpy.ndarray([-1.0, 1.0])).
    azi : array_like[float], optional
        Angles that demarcate azimuthal angular tally bins (default numpy.ndarray([-1.0, 1.0])).
    g : array_like[float], optional
        Energies that demarcate energy tally bins (default numpy.ndarray([-INF, INF])).
    E : array_like[float], optional
        Continuous energy functionality, (default numpy.ndarray([0.0, INF])).

    Returns
    -------
    dictionary
        A tally card.
    """

    # Get tally card
    card = global_.input_deck.tally

    # Set mesh
    card["mesh"]["x"] = x
    card["mesh"]["y"] = y
    card["mesh"]["z"] = z
    card["mesh"]["t"] = t
    card["mesh"]["mu"] = mu
    card["mesh"]["azi"] = azi

    # Set energy group grid
    if type(g) == type("string") and g == "all":
        G = global_.input_deck.materials[0].G
        card["mesh"]["g"] = np.linspace(0, G, G + 1) - 0.5
    else:
        card["mesh"]["g"] = g
    if global_.input_deck.setting["mode_CE"]:
        card["mesh"]["g"] = E

    return card
