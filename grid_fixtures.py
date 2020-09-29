"""
Fixtures with pre-defined lattices to be tested.
"""

import numpy as np

# crossroad with one-way roads
grid3x3 = np.array([
    [
        {'cell_type': 'block'},
        {'cell_type': 'road', 'allowed_directions': ['east']},
        {'cell_type': 'block'}
    ],

    [
        {'cell_type': 'road', 'allowed_directions': ['north']},
        {'cell_type': 'road', 'allowed_directions': ['east', 'north']},
        {'cell_type': 'road', 'allowed_directions': ['east']},
    ],

    [
        {'cell_type': 'block'},
        {'cell_type': 'road', 'allowed_directions': ['east']},
        {'cell_type': 'block'}
    ],

])

# Two-way crossroad
grid6x6 = np.array([
    [
        {'cell_type': 'block'},
        {'cell_type': 'block'},
        {'cell_type': 'road', 'allowed_directions': ['east']},
        {'cell_type': 'road', 'allowed_directions': ['west']},
        {'cell_type': 'block'},
        {'cell_type': 'block'}
    ],

    [
        {'cell_type': 'block'},
        {'cell_type': 'block'},
        {'cell_type': 'road', 'allowed_directions': ['east']},
        {'cell_type': 'road', 'allowed_directions': ['west']},
        {'cell_type': 'block'},
        {'cell_type': 'block'}
    ],

    [
        {'cell_type': 'road', 'allowed_directions': ['south']},
        {'cell_type': 'road', 'allowed_directions': ['south']},
        {'cell_type': 'road', 'allowed_directions': ['south', 'east']},
        {'cell_type': 'road', 'allowed_directions': ['south', 'west']},
        {'cell_type': 'road', 'allowed_directions': ['south']},
        {'cell_type': 'road', 'allowed_directions': ['south']},
    ],

    [
        {'cell_type': 'road', 'allowed_directions': ['north']},
        {'cell_type': 'road', 'allowed_directions': ['north']},
        {'cell_type': 'road', 'allowed_directions': ['north', 'east']},
        {'cell_type': 'road', 'allowed_directions': ['north', 'west']},
        {'cell_type': 'road', 'allowed_directions': ['north']},
        {'cell_type': 'road', 'allowed_directions': ['north']},
    ],

    [
        {'cell_type': 'block'},
        {'cell_type': 'block'},
        {'cell_type': 'road', 'allowed_directions': ['east']},
        {'cell_type': 'road', 'allowed_directions': ['west']},
        {'cell_type': 'block'},
        {'cell_type': 'block'}
    ],

    [
        {'cell_type': 'block'},
        {'cell_type': 'block'},
        {'cell_type': 'road', 'allowed_directions': ['east']},
        {'cell_type': 'road', 'allowed_directions': ['west']},
        {'cell_type': 'block'},
        {'cell_type': 'block'}
    ],

])

grid12x6 = np.concatenate((grid6x6, grid6x6))
