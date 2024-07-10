from CamFunctions import *
from EventFunctions import *
import pandas as pd


# Spawning all cameras
def spawn_three_winged_station(
    wing_dimension: float,
    telephoto_camera={
        "count": int,
        "distribution": float,
        "max_range": float,
        "gimbal_limit": float,
        "gimbal_rate_limit": float,
        "min_luminosity": float,
        "focal_length": float,
        "fov": tuple,
    },
    mediumrange_camera={
        "count": int,
        "distribution": float,
        "max_range": float,
        "gimbal_limit": float,
        "gimbal_rate_limit": float,
        "min_luminosity": float,
        "focal_length": float,
        "fov": tuple,
    },
    wideangle_camera={
        "count": int,
        "distribution": float,
        "max_range": float,
        "gimbal_limit": float,
        "gimbal_rate_limit": float,
        "min_luminosity": float,
        "focal_length": float,
        "fov": tuple,
    },
):
    """
    This function spawns a set of cameras in 3 wings.
    The count parameter is the number of cameras per wing. So the total number of cameras will be multiplied by 3.
    The distribution parameter is the radius of the circle on which the cameras will be evenly distributed per wing.
    To modify some of the other parameters of the cameras themselves, please modify the parameters at Camera.py
    Default keyword arguments (kwargs) dont work inside a dictionary, so you will have to specify ALL the key value pairs in the dictionaries.
    """

    list_of_cameras = []
    list_of_types = []
    list_of_coordinates = []
    list_of_fovs = []
    list_of_pointings = []
    list_of_up_vectors = []
    list_of_gimbal_limits = []
    list_of_maxranges = []
    list_of_luminosities = []
    list_of_wingnumbers = []

    for wing_number in [1, 2, 3]:
        coordinates = [0.0, 0.0, 0.0]
        pointing = [0.0, 0.0, 0.0]

        # The x, or y, or z coordinates of the cameras are changed to (dimension/2) based on the wing. Wing 1 is shifted in x, 2 in y, 3 in z #
        coordinates[wing_number - 1] = wing_dimension / 2

        circle_center = np.array(coordinates).copy()

        for i in range(telephoto_camera["count"]):
            if wing_number == 1:
                coordinates[1] = telephoto_camera["distribution"] * np.cos(
                    2 * np.pi * i / telephoto_camera["count"]
                )
                coordinates[2] = telephoto_camera["distribution"] * np.sin(
                    2 * np.pi * i / telephoto_camera["count"]
                )
            elif wing_number == 2:
                coordinates[0] = telephoto_camera["distribution"] * np.cos(
                    2 * np.pi * i / telephoto_camera["count"]
                )
                coordinates[2] = telephoto_camera["distribution"] * np.sin(
                    2 * np.pi * i / telephoto_camera["count"]
                )
            elif wing_number == 3:
                coordinates[0] = telephoto_camera["distribution"] * np.cos(
                    2 * np.pi * i / telephoto_camera["count"]
                )
                coordinates[1] = telephoto_camera["distribution"] * np.sin(
                    2 * np.pi * i / telephoto_camera["count"]
                )

            # Need to try out different pointings
            pointing = np.array(coordinates) / np.linalg.norm(np.array(coordinates))
            up = np.cross((np.array(coordinates) - circle_center), pointing)

            cam = make_camera(
                type="telephoto",
                coordinates=coordinates,
                pointing_vector=pointing,
                up_vector=up,
            )

            cam.name = (
                f"telephoto camera number {i+1} in wing {wing_number} at {coordinates}"
            )
            cam.gimbal_limit = telephoto_camera["gimbal_limit"]
            cam.gimbal_rate_limit = telephoto_camera["gimbal_rate_limit"]
            cam.max_range = telephoto_camera["max_range"]
            cam.min_luminosity = telephoto_camera["min_luminosity"]
            cam.focal_length = telephoto_camera["focal_length"]

            cam.fov = telephoto_camera["fov"]
            cam.get_fov_bounding_vectors()

            initial_pointing = [0.0, 0.0, 0.0]
            # "Initial Pointing" of wing 1 is set to 1, ie., wing 1 looks at x-axis, 2 at y-axis, 3 at z-axis #
            initial_pointing[wing_number - 1] = 1.0

            cam.inertial_frame_initial_pointing = np.array(initial_pointing)

            list_of_cameras.append(cam)

            list_of_types.append(cam.type)
            list_of_wingnumbers.append(wing_number)
            list_of_coordinates.append(cam.inertial_coordinates)
            list_of_fovs.append(cam.fov)
            list_of_pointings.append(cam.inertial_frame_pointing_vector)
            list_of_up_vectors.append(cam.inertial_frame_up_vector)
            list_of_gimbal_limits.append(cam.gimbal_limit)
            list_of_maxranges.append(cam.max_range)
            list_of_luminosities.append(cam.min_luminosity)

        for j in range(mediumrange_camera["count"]):
            if wing_number == 1:
                coordinates[1] = mediumrange_camera["distribution"] * np.cos(
                    2 * np.pi * j / mediumrange_camera["count"]
                )
                coordinates[2] = mediumrange_camera["distribution"] * np.sin(
                    2 * np.pi * j / mediumrange_camera["count"]
                )
            elif wing_number == 2:
                coordinates[0] = mediumrange_camera["distribution"] * np.cos(
                    2 * np.pi * j / mediumrange_camera["count"]
                )
                coordinates[2] = mediumrange_camera["distribution"] * np.sin(
                    2 * np.pi * j / mediumrange_camera["count"]
                )
            elif wing_number == 3:
                coordinates[0] = mediumrange_camera["distribution"] * np.cos(
                    2 * np.pi * j / mediumrange_camera["count"]
                )
                coordinates[1] = mediumrange_camera["distribution"] * np.sin(
                    2 * np.pi * j / mediumrange_camera["count"]
                )

            pointing = np.array(coordinates) / np.linalg.norm(np.array(coordinates))
            up = np.cross((np.array(coordinates) - circle_center), pointing)

            cam = make_camera(
                type="medium range",
                coordinates=coordinates,
                pointing_vector=pointing,
                up_vector=up,
            )

            cam.name = f"medium range camera number {j+1} in wing {wing_number} at {coordinates}"
            cam.gimbal_limit = mediumrange_camera["gimbal_limit"]
            cam.gimbal_rate_limit = mediumrange_camera["gimbal_rate_limit"]
            cam.max_range = mediumrange_camera["max_range"]
            cam.min_luminosity = mediumrange_camera["min_luminosity"]
            cam.focal_length = mediumrange_camera["focal_length"]

            cam.fov = mediumrange_camera["fov"]
            cam.get_fov_bounding_vectors()

            initial_pointing = [0.0, 0.0, 0.0]
            # "Initial Pointing" of wing 1 is set to 1, ie., wing 1 looks at x-axis, 2 at y-axis, 3 at z-axis #
            initial_pointing[wing_number - 1] = 1.0

            cam.inertial_frame_initial_pointing = np.array(initial_pointing)

            list_of_cameras.append(cam)

            list_of_cameras.append(cam)
            list_of_types.append(cam.type)
            list_of_wingnumbers.append(wing_number)
            list_of_coordinates.append(cam.inertial_coordinates)
            list_of_fovs.append(cam.fov)
            list_of_pointings.append(cam.inertial_frame_pointing_vector)
            list_of_up_vectors.append(cam.inertial_frame_up_vector)
            list_of_gimbal_limits.append(cam.gimbal_limit)
            list_of_maxranges.append(cam.max_range)
            list_of_luminosities.append(cam.min_luminosity)

        for k in range(wideangle_camera["count"]):
            if wing_number == 1:
                coordinates[1] = wideangle_camera["distribution"] * np.cos(
                    2 * np.pi * k / wideangle_camera["count"]  # type: ignore
                )
                coordinates[2] = wideangle_camera["distribution"] * np.sin(
                    2 * np.pi * k / wideangle_camera["count"]  # type: ignore
                )
            elif wing_number == 2:
                coordinates[0] = wideangle_camera["distribution"] * np.cos(
                    2 * np.pi * k / wideangle_camera["count"]
                )
                coordinates[2] = wideangle_camera["distribution"] * np.sin(
                    2 * np.pi * k / wideangle_camera["count"]
                )
            elif wing_number == 3:
                coordinates[0] = wideangle_camera["distribution"] * np.cos(
                    2 * np.pi * k / wideangle_camera["count"]
                )
                coordinates[1] = wideangle_camera["distribution"] * np.sin(
                    2 * np.pi * k / wideangle_camera["count"]
                )

            pointing = np.array(coordinates) / np.linalg.norm(np.array(coordinates))
            up = np.cross((np.array(coordinates) - circle_center), pointing)

            cam = make_camera(
                type="wide angle",
                coordinates=coordinates,
                pointing_vector=pointing,
                up_vector=up,
            )

            cam.name = (
                f"wide angle camera number {k+1} in wing {wing_number} at {coordinates}"
            )

            cam.gimbal_limit = wideangle_camera["gimbal_limit"]
            cam.gimbal_rate_limit = wideangle_camera["gimbal_rate_limit"]
            cam.max_range = wideangle_camera["max_range"]
            cam.min_luminosity = wideangle_camera["min_luminosity"]
            cam.focal_length = wideangle_camera["focal_length"]

            cam.fov = wideangle_camera["fov"]
            cam.get_fov_bounding_vectors()

            initial_pointing = [0.0, 0.0, 0.0]
            # "Initial Pointing" of wing 1 is set to 1, ie., wing 1 looks at x-axis, 2 at y-axis, 3 at z-axis #
            initial_pointing[wing_number - 1] = 1.0

            cam.inertial_frame_initial_pointing = np.array(initial_pointing)

            list_of_cameras.append(cam)

            list_of_cameras.append(cam)
            list_of_types.append(cam.type)
            list_of_wingnumbers.append(wing_number)
            list_of_coordinates.append(cam.inertial_coordinates)
            list_of_fovs.append(cam.fov)
            list_of_pointings.append(cam.inertial_frame_pointing_vector)
            list_of_up_vectors.append(cam.inertial_frame_up_vector)
            list_of_gimbal_limits.append(cam.gimbal_limit)
            list_of_maxranges.append(cam.max_range)
            list_of_luminosities.append(cam.min_luminosity)

    dictionary = {
        "Camera Type": list_of_types,
        "Wing Number": list_of_wingnumbers,
        "Coordinates": list_of_coordinates,
        "FoV": list_of_fovs,
        "Pointing Vector": list_of_pointings,
        "Up Vector": list_of_up_vectors,
        "Camera Gimbal Limit": list_of_gimbal_limits,
        "Max. Range (m)": list_of_maxranges,
        "Min. Luminosity": list_of_luminosities,
    }
    dataframe = pd.DataFrame(dictionary)

    return (list_of_cameras, dataframe)


# Spawning Events
def spawn_random_events(
    no_of_events: int,
    x_y_z_extents_low,
    x_y_z_extents_high,
    static_movement=True,
    static_luminosity=True,
    velocity_limit=(-10.0, 10),
    luminosity_limit=(-0.5, 1.0),
    luminosity_rate_limit=(-0.01, 0.01),
):
    """
    This function spawns random events with randomized velocities and luminosities and luminosity rates over the given x,y,z extents.
    The x_y_z_extents should be an array of length 3.
    The events are all static in terms of velocity and luminosity by default.
    For time varying velocities, you will need to modify the parameters of the events in the simulation.
    """
    list_of_events = []

    for i in range(no_of_events):
        x_coordinate = np.random.uniform(
            low=x_y_z_extents_low[0], high=x_y_z_extents_high[0]
        )
        y_coordinate = np.random.uniform(
            low=x_y_z_extents_low[1], high=x_y_z_extents_high[1]
        )
        z_coordinate = np.random.uniform(
            low=x_y_z_extents_low[2], high=x_y_z_extents_high[2]
        )

        # Change the below "low" value if you don't want events to spawn invisible
        luminosity = np.random.uniform(
            low=luminosity_limit[0], high=luminosity_limit[1]
        )

        if static_movement:
            x_velocity = 0.0
            y_velocity = 0.0
            z_velocity = 0.0
        else:
            x_velocity = np.random.uniform(
                low=velocity_limit[0], high=velocity_limit[1]
            )
            y_velocity = np.random.uniform(
                low=velocity_limit[0], high=velocity_limit[1]
            )
            z_velocity = np.random.uniform(
                low=velocity_limit[0], high=velocity_limit[1]
            )

        if static_luminosity:
            luminosity_rate = 0.0
        else:
            luminosity_rate = np.random.uniform(
                low=luminosity_rate_limit[0], high=luminosity_rate_limit[1]
            )

        eve = make_event(
            coordinates=[x_coordinate, y_coordinate, z_coordinate],
            luminosity=luminosity,
            inertial_frame_linear_velocity=[x_velocity, y_velocity, z_velocity],
            luminosity_rate=luminosity_rate,
        )
        eve.name = f"Event_{i+1}"

        list_of_events.append(eve)

    return list_of_events


def generate_events_config(
    no_of_events,
    x_y_z_extents,
):
    pass


def spawn_events_from_config(config_dataframe):
    pass
