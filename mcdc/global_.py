from mcdc.class_.global_ import Global

global_            = Global()
population_control = None

# We separate population_control as it is pure Python (not Numba)
