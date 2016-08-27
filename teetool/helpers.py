# support functions

import colorsys
import numpy as np

def getDistinctColours(ncolours):
    """
    returns N distinct colors using the colourspace.
    spreads equally in hue space, then converts to RGB
    """
    # check
    if type(ncolours) is not int:
        raise TypeError("expected integer, not {0}".format(type(ncolours)))

    if (ncolours < 1):
        raise ValueError("expected integer to be larger than 0, not {0}".format(ncolours))

    # spread equally in hue space
    HSV_tuples = [(x*1.0/ncolours, 0.5, 0.5) for x in range(ncolours)]

    # conver to RGB
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    return list(RGB_tuples)
