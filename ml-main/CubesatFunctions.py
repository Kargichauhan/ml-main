from Cubesat import *


def make_sat(size: str, origin, orientation=R.identity()):
    """
    The "size" argument should be a string ("1U" or "3U"),
    the "origin" argument should have 3 elements (array like),
    the "orientation" argument should be a "Rotation" instance from scipy.spatial.transform
    """

    sat = cubesat(size)

    sat.translate(origin)
    sat.rotate(orientation)  # USES q-convention!!! Beware!!! #

    sat.get_vertices()
    sat.get_edges()
    sat.get_planes()
    sat.get_leds()

    return sat
