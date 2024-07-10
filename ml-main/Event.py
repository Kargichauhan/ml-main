from scipy.spatial.transform import Rotation as R  # type: ignore
import numpy as np


class event:
    def __init__(self, inertial_coordinates):
        """
        Generates an "Event" at a particular coordinate in the inertial frame.
        The velocity is (0,0,0) (static event) by default.
        The luminosity is 1 (fully bright) by default and the rate of change of luminosity is 0 (unchanging event).
        The make_event() function at EventFunctions.py will be used to fully initialize an Event.
        """

        self.name = "default static event"

        self.inertial_coordinates = np.array(inertial_coordinates).reshape((3,))
        self.inertial_frame_linear_velocity = np.array([0.0, 0.0, 0.0])

        self.luminosity = 1.0
        self.luminosity_rate = 0.0

        self.cameras_that_can_see = []
        self.camera_assigned = {
            "telephoto": None,
            "medium range": None,
            "wide angle": None,
        }
        # A dictionary of camera objects assigned by "assign_events_to_cameras" function

    def translate(self, distance):
        """
        This method moves the event by "distance".
        It also stores the cumulative distance moved by the camera in all three coordinates.
        The "distance" input argument should be a 3-array like.
        """

        distance = np.array(distance).reshape((3,))

        self.inertial_coordinates = (
            self.inertial_coordinates + distance
        )  # Adding new translation to the previous distance to keep track #

    def move_to(self, inertial_coordinates):
        """
        This method just moves the event object to a particular coordinate instead of translating it.
        """
        self.inertial_coordinates = np.array(inertial_coordinates).reshape(
            3,
        )
