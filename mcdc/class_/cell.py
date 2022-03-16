class Cell:
    def __init__(self, ID, surfaces, senses, material):
        self.ID         = ID
        self.surfaces   = surfaces
        self.senses     = senses
        self.material   = material
        self.n_surfaces = len(surfaces)
    
    # Test if position pos is inside the cell
    def test_point(self, P):
        for i in range(self.n_surfaces):
            if self.surfaces[i].evaluate(P) * self.senses[i] < 0.0:
                return False
        return True
