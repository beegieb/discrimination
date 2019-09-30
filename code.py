import numpy as np
import random

from itertools import product


class GridEnvironment:
    def __init__(self, dimensions, wrap=True):
        assert len(dimensions) == 2
        assert dimensions[0] > 0
        assert dimensions[1] > 0
        
        self._dimensions = dimensions
        self._grid = np.zeros(dimensions)
        self._unoccupied = set(
            product(range(dimensions[0]), 
                    range(dimensions[1]))
        )
        self._occupied = {}
        self._wrap = wrap
    
    @property
    def dimensions(self):
        return self._dimensions
    
    @property
    def wrap(self):
        return self._wrap
    
    def is_empty(self, grid_pos):
        return grid_pos in self._unoccupied
    
    def is_occupied(self, grid_pos):
        return not self.is_empty(grid_pos)
    
    def add_occupant(self, occupant, grid_pos=None):
        """
        Adds an occupant to grid_pos
        
        If grid_pos is None a random unoccupied position is selected
        
        Returns the grid pos of the occupant
        """
        if grid_pos is not None and self.is_occupied(grid_pos):
            raise ValueError("Provided grid pos {} is currently occupied")
        
        if grid_pos is None:
            grid_pos = self.get_unoccupied_pos()
        
        self._occupied[grid_pos] = occupant
        self._unoccupied.discard(grid_pos)
        
        return grid_pos
    
    def remove_occupant(self, grid_pos):
        """
        Remove an occupant from grid_pos
        
        Returns the occupant 
        """
        if self.is_empty(grid_pos):
            raise ValueError("Provided grid pos {} is currently empty")
        
        occupant = self._occupied.pop(grid_pos)
        
        self._unoccupied.add(grid_pos)
        
        return occupant
    
    def get_unoccupied(self):
        return self._unoccupied
    
    def get_unoccupied_pos(self):
        """Returns an unoccupied position on the grid"""
        return random.choice(list(self._unoccupied))
    
    def get_neighbourhood(self, grid_pos, neighbourhood_radius=1):
        """Returns all occpuants in the neighbourhood centered at grid_pos"""
        
        r = neighbourhood_radius
        x, y = grid_pos
        
        dx, dy = self.dimensions 
        if self.wrap:
            xs = np.arange(x - r, x + r + 1) % dx
            ys = np.arange(y - r, y + r + 1) % dy
        else:
            xs = np.arange(max(0, x - r), min(dx, x + r + 1))
            ys = np.arange(max(0, y - r), min(dy, y + r + 1))
        
        return {
            pos: self._occupied[pos]
            for pos in product(xs, ys) 
            if self.is_occupied(pos)
        }
    
    def get_grid(self, value_fn=None):
        if value_fn is None:
            value_fn = lambda x: x
            
        grid = np.zeros(self.dimensions)
        for xi, yi in self._occupied:
            grid[xi, yi] = value_fn(self._occupied[xi, yi])
        
        return grid
        


class GenericOccupant:
    def __init__(self, properties, pos=None):
        self.validate_properties(properties)
        self._properties = properties
        self._pos = pos
    
    def __repr__(self):
        return '{s.__class__.__name__}(properties={s.properties})'.format(s=self)
    
    @property
    def pos(self):
        return self._pos
    
    @pos.setter
    def pos(self, new_pos):
        assert len(new_pos) == 2
        assert new_pos[0] >= 0
        assert new_pos[1] >= 0
        self._pos = new_pos
    
    @property
    def properties(self):
        return self._properties
    
    @classmethod
    def validate_properties(cls, properties):
        """
        raise ValueError if properties are not valid
        """
        return cls._validate_properties(properties)
    
    def score_neighbourhood(self, neighbourhood):
        """
        Returns a numeric score of a neighbourhood indicating the occupants affinity
        """
        raise NotImplementedError('Not Implemented')
    
    def select_neighbourhood(self, neighbourhood):
        """
        Return True if the occupant would select the neighbourhood, False otherwise
        """
        raise NotImplementedError('Not Implemented')
        


class SchellingOccupant(GenericOccupant):
    """
    Properties should include
    
    identity: the identity of the occupant (integer)
    F: the F-value of the Schelling Model corresponding to the occupants affinity towards 
       neighbours of the same identity
    """
    
    @staticmethod
    def _validate_properties(properties):
        try:
            identity = properties['identity']
        except KeyError:
            raise ValueError('identity not found in properties')
        
        try:
            F = properties['F']
        except KeyError:
            raise ValueError('F not found in properties')
            
        try:
            assert(0 < F < 1)
        except AssertionError:
            raise ValueError('F should be strictly between 0 and 1')
            
    def score_neighbourhood(self, neighbourhood):
        size = len(neighbourhood)
        if size > 0:
            total = np.sum([n.properties['identity'] == self.properties['identity'] 
                            for n in neighbourhood.values()])
            return (total - 1) / size
        else:
            return 0
    
    def select_neighbourhood(self, neighbourhood):
        return self.score_neighbourhood(neighbourhood) > self.properties['F']
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    random.seed(1234)

    cmap = 'Set1'

    n_groups = 2
    F = 3/8
    Fs = [F] * n_groups

    d = 100
    dx, dy = (d, d)

    n = int(d * d / (n_groups + 2))

    occupants = [
        SchellingOccupant({'F': Fs[i % n_groups], 'identity': (i % n_groups) + 1})
        for i in range(n * n_groups)
    ]

    env = GridEnvironment((dx, dy), wrap=True)
    for o in occupants:
        o.pos = env.add_occupant(o)
        
    maxiter = 10

    for i in range(maxiter):
        n_moves = 0
        for o in occupants:
            n = env.get_neighbourhood(o.pos, neighbourhood_radius=1)
            if not o.select_neighbourhood(n):
                old_pos = o.pos
                env.remove_occupant(old_pos)
                found_new = False

                for new_pos in env.get_unoccupied():
                    n = env.get_neighbourhood(new_pos)

                    if o.select_neighbourhood(n):
                        found_new = True
                        o.pos = new_pos
                        env.add_occupant(o, new_pos)
                        n_moves += 1
                        break

                if not found_new:
                    o.pos = old_pos
                    env.add_occupant(o, old_pos)
        if n_moves == 0:
            break
    plt.imshow(env.get_grid(value_fn=lambda x: x.properties['identity']), cmap=cmap)
