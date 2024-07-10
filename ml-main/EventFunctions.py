from Event import *


def make_event(
    coordinates,
    luminosity: float,
    inertial_frame_linear_velocity,
    luminosity_rate: float,
):
    event1 = event(coordinates)
    event1.luminosity = luminosity
    event1.luminosity_rate = luminosity_rate
    event1.inertial_frame_linear_velocity = np.array(inertial_frame_linear_velocity)
    return event1
