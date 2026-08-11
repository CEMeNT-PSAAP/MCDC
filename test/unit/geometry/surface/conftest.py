import pytest


@pytest.fixture
def compile_surfaces(prepare_simulation):
    def _compile(static_surface_obj, moving_surface_obj):
        structure_container, data = prepare_simulation(
            objects=[static_surface_obj, moving_surface_obj]
        )
        structure = structure_container[0]
        static_surface = structure["surfaces"][static_surface_obj.ID]
        moving_surface = structure["surfaces"][moving_surface_obj.ID]
        return static_surface, moving_surface, data

    return _compile
