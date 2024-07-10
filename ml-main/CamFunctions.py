from Camera import *
from Cubesat import *
from Event import *
from scipy.spatial.transform import Rotation as R
from scipy.optimize import nnls


def make_camera(type: str, coordinates, pointing_vector, up_vector):
    """
    The pointing vector and the up vector must be mutually orthogonal, otherwise there will be an error.
    The coordinates, pointing and up vectors should be in the inertial frame.
    """
    cam = camera(type)
    cam.translate(coordinates)
    cam.point_to(pointing_vector, up_vector)
    cam.inertial_frame_initial_pointing = np.array(pointing_vector)

    return cam


def fov_test(camera_object, event_object):
    """
    This function does linear programming to determine if the Event Object is present within the Inertial Frame FoV vectors of the Camera Object.
    Using this algorithm: https://math.stackexchange.com/q/2144085
    """

    camera_object.event_assigned = None
    event_object.camera_assigned = {
        "telephoto": None,
        "medium range": None,
        "wide angle": None,
    }
    # The assigning can be refreshed every time the function is called, because the assigning will happen after all the visibility tests are done separately in each timestep.

    (coefficients, residuals) = nnls(
        camera_object.inertial_frame_fov_bounding_vectors.T,
        (event_object.inertial_coordinates - camera_object.inertial_coordinates),
    )
    # If there are residuals, then there is an error in the results. I am observing errors only in false positives so far.
    if residuals > 0:
        if camera_object in event_object.cameras_that_can_see:
            event_object.cameras_that_can_see.remove(camera_object)
        if event_object in camera_object.events_visible:
            camera_object.events_visible.remove(event_object)
        # if event_object is camera_object.event_assigned:
        #     camera_object.event_assigned = None
        # if camera_object is event_object.camera_assigned[camera_object.type]:
        #     event_object.camera_assigned[camera_object.type] = None
        return False
    # All coefficients being zero means the event coincides with the camera.
    elif all(coefficients == 0):
        if camera_object in event_object.cameras_that_can_see:
            event_object.cameras_that_can_see.remove(camera_object)
        if event_object in camera_object.events_visible:
            camera_object.events_visible.remove(event_object)
        # if event_object is camera_object.event_assigned:
        #     camera_object.event_assigned = None
        # if camera_object is event_object.camera_assigned[camera_object.type]:
        #     event_object.camera_assigned[camera_object.type] = None
        return False
    elif all(coefficients >= 0):
        if camera_object not in event_object.cameras_that_can_see:
            event_object.cameras_that_can_see.append(camera_object)
        if event_object not in camera_object.events_visible:
            camera_object.events_visible.append(event_object)
        return True
    else:
        if camera_object in event_object.cameras_that_can_see:
            event_object.cameras_that_can_see.remove(camera_object)
        if event_object in camera_object.events_visible:
            camera_object.events_visible.remove(event_object)
        # if event_object is camera_object.event_assigned:
        #     camera_object.event_assigned = None
        # if camera_object is event_object.camera_assigned[camera_object.type]:
        #     event_object.camera_assigned[camera_object.type] = None
        return False


def los_test(camera_object):
    """
    This function applies a manual depth buffer to find out what all Events in the Camera's FoV are directly in its LoS.
    For point events, the algorithm is simple, we iterate through all events in a Camera's FoV, then find out if the vectors connecting the camera and the event are collinear.
    Collinearity check is done by taking cross product of two vectors and if they are collinear, the cross product is zero.
    After the collinearity check is done for all combinations of vectors, the vector which has the smallest magnitude among the collinear vectors is the visible one, and the others are removed from the list.
    For Objects, the check first involves finding the point of intersection between the vector and the planes of a body.
    The algorithm for plates, cylinders, cones, and spheres are given in the PowerPoint presentation.
    """

    if len(camera_object.events_visible) == 0:
        return

    list_of_CamToEvent_vectors = []

    for event_object in camera_object.events_visible:
        CamToEvent = (
            event_object.inertial_coordinates - camera_object.inertial_coordinates
        )
        list_of_CamToEvent_vectors.append(
            (event_object, CamToEvent, np.linalg.norm(CamToEvent))
        )

    for i in range(len(list_of_CamToEvent_vectors)):
        for j in range(len(list_of_CamToEvent_vectors)):
            cross_product = np.cross(
                list_of_CamToEvent_vectors[i][1], list_of_CamToEvent_vectors[j][1]
            )

            if i == j:  # crossing the vector with itself
                continue

            elif (
                camera_object
                not in list_of_CamToEvent_vectors[i][0].cameras_that_can_see
            ):  # if the camera object is not in the list of cameras that can see that event, ie., it got removed already
                continue

            elif (
                camera_object
                not in list_of_CamToEvent_vectors[j][0].cameras_that_can_see
            ):  # if the camera object is not in the list of cameras that can see that event, ie., it got removed already
                continue

            elif (
                (cross_product[0] == 0)
                & (cross_product[1] == 0)
                & (cross_product[2] == 0)
            ):  # Ridiculous way of checking if the cross product is zero!
                if list_of_CamToEvent_vectors[i][2] <= list_of_CamToEvent_vectors[j][2]:
                    list_of_CamToEvent_vectors[j][0].cameras_that_can_see.remove(
                        camera_object
                    )
                    camera_object.events_visible.remove(
                        list_of_CamToEvent_vectors[j][0]
                    )
                elif (
                    list_of_CamToEvent_vectors[j][2] < list_of_CamToEvent_vectors[i][2]
                ):
                    list_of_CamToEvent_vectors[i][0].cameras_that_can_see.remove(
                        camera_object
                    )
                    camera_object.events_visible.remove(
                        list_of_CamToEvent_vectors[i][0]
                    )


def luminosity_test(camera_object):
    temp_list = camera_object.events_visible.copy()
    # Need to make a .copy() of the original list of vectors, as python will changes to the original list (dynamic allocation), and that messes up the loop
    for event_object in temp_list:
        if event_object.luminosity < camera_object.min_luminosity:
            camera_object.events_visible.remove(event_object)
            event_object.cameras_that_can_see.remove(camera_object)


def range_test(camera_object):
    temp_list = camera_object.events_visible.copy()
    # Need to make a .copy() of the original list of vectors, as python will changes to the original list (dynamic allocation), and that messes up the loop
    for event_object in temp_list:
        if np.linalg.norm(event_object.inertial_coordinates) > camera_object.max_range:
            camera_object.events_visible.remove(event_object)
            event_object.cameras_that_can_see.remove(camera_object)


def fov_led_test(camera_object, cubesat_objects_list):
    """
    This function does linear programming to determine if the LEDs of a CubeSat are present within the Inertial Frame FoV vectors of the Camera Object.
    Using this algorithm: https://math.stackexchange.com/q/2144085
    """
    # Firstly, we have to refresh the list every time we do the test. To prevent residue from the previous test.
    camera_object.leds_visible = []

    for cubesat_object in cubesat_objects_list:
        leds = cubesat_object.leds_list.copy()

        for led in leds:
            # Firstly, we have to refresh the list every time we do the test. To prevent residue from the previous test.
            if camera_object in led["cameras_that_can_see"]:
                led["cameras_that_can_see"].remove(camera_object)

            (coefficients, residuals) = nnls(
                camera_object.inertial_frame_fov_bounding_vectors.T,
                (led["inertial_coordinate"] - camera_object.inertial_coordinates),
            )
            # If there are residuals, then there is an error in the results. I am observing errors only in false positives so far.
            if residuals > 0:
                if camera_object in led["cameras_that_can_see"]:
                    led["cameras_that_can_see"].remove(camera_object)

                # Stupid way of checking if the led object (which is a dictionary) is inside the camera_object.leds_visible list
                index_length = len(camera_object.leds_visible)
                for index in range(index_length):
                    if camera_object.leds_visible[index] is led:
                        del camera_object.leds_visible[index]
                        break

                if cubesat_object in camera_object.cubesats_in_fov:
                    camera_object.cubesats_in_fov.remove(cubesat_object)

            # All coefficients being zero means the event coincides with the camera.
            elif all(coefficients == 0):
                if camera_object in led["cameras_that_can_see"]:
                    led["cameras_that_can_see"].remove(camera_object)

                # Stupid way of checking if the led object (which is a dictionary) is inside the camera_object.leds_visible list
                index_length = len(camera_object.leds_visible)
                for index in range(index_length):
                    if camera_object.leds_visible[index] is led:
                        del camera_object.leds_visible[index]
                        break

                if cubesat_object in camera_object.cubesats_in_fov:
                    camera_object.cubesats_in_fov.remove(cubesat_object)

            # FoV test pass
            elif all(coefficients >= 0):
                if camera_object not in led["cameras_that_can_see"]:
                    led["cameras_that_can_see"].append(camera_object)

                # Extremely stupid way of checking if the led object (which is a dictionary) is NOT inside the camera_object.leds_visible list
                flags = []
                for dictionary in camera_object.leds_visible:
                    if dictionary is led:
                        flags.append(True)
                # if flag is appended with True, that means the led (dictionary) is in the camera_object.leds_visible list
                if not any(flags):
                    camera_object.leds_visible.append(led)

                if cubesat_object not in camera_object.cubesats_in_fov:
                    camera_object.cubesats_in_fov.append(cubesat_object)

            # Other failure conditions
            else:
                if camera_object in led["cameras_that_can_see"]:
                    led["cameras_that_can_see"].remove(camera_object)

                # Stupid way of checking if the led object (which is a dictionary) is inside the camera_object.leds_visible list
                index_length = len(camera_object.leds_visible)
                for index in range(index_length):
                    if camera_object.leds_visible[index] is led:
                        del camera_object.leds_visible[index]
                        break

                if cubesat_object in camera_object.cubesats_in_fov:
                    camera_object.cubesats_in_fov.remove(cubesat_object)


def los_led_test(camera_object):
    """
    This function does does line of sight analysis between LEDs of a CubeSat and its planes with respect to a camera.
    It does the fov analysis using four points of a face using the same algorithm as present in the previous function. The test passes if the LED is present outside the orthant.
    If the LED is present inside the orthant, then the point of intersection between the planes and the connecting line is found using this algorithm: https://math.stackexchange.com/q/100447
    If the point is nearer, then the LED is being blocked by a parallel face.
    """
    leds = camera_object.leds_visible.copy()
    camera_coordinates = camera_object.inertial_coordinates

    # Getting a list of all CubeSats that are currently seen by the camera
    cubesat_list = []
    for led in leds:
        if led["parent_cubesat"] in cubesat_list:
            continue
        else:
            cubesat_list.append(led["parent_cubesat"])

    # Getting a list of all planes of all CubeSats involved
    planes_list = []
    for cubesat_object in cubesat_list:
        planes_list.extend(cubesat_object.planes_list.copy())

    for led in leds:
        # fov test has failed already
        if camera_object not in led["cameras_that_can_see"]:
            continue

        connecting_line = led["inertial_coordinate"] - camera_coordinates
        connecting_direction = connecting_line / np.linalg.norm(connecting_line)

        for plane in planes_list:
            # skip the plane that the led is on firstly
            if (
                plane["name"] == led["plane"]["name"]
                and plane["parent_cubesat"] is led["parent_cubesat"]
            ):
                continue

            # do the fov test to see if the led is within the orthant
            plane_points = []
            plane_point_vectors = []
            for point in plane["points"]:
                plane_points.append(point["inertial_coordinate"])
                vector = point["inertial_coordinate"] - np.array(camera_coordinates)
                plane_point_vectors.append(vector)

            plane_point_vectors = np.array(plane_point_vectors)
            (coefficients, residuals) = nnls(
                plane_point_vectors.T,
                connecting_line,
            )

            # residuals > 0 means the connecting_line vector is outside the orthant, so it passes the test
            if residuals > 0:
                continue
            # test fails if the camera and the led coincide
            elif all(coefficients == 0):
                if camera_object in led["cameras_that_can_see"]:
                    led["cameras_that_can_see"].remove(camera_object)

                # Stupid way of checking if the led object (which is a dictionary) is inside the camera_object.leds_visible list
                index_length = len(camera_object.leds_visible)
                for index in range(index_length):
                    if camera_object.leds_visible[index] is led:
                        del camera_object.leds_visible[index]
                        break

                break  # Break statement added because we do not have to test the led anymore

            # if the connecting_line vector lies inside the orthant, then we will have to do another test to see the los test passes.
            # there are cases where the fov test passes but the led may still be blocked (parallel faces)
            elif all(coefficients >= 0):
                normal_vector = plane["plane_inertial_coordinate"]

                plane_point = plane_points[0]

                t = -(
                    np.dot(normal_vector, (camera_coordinates - plane_point))
                ) / np.dot(normal_vector, connecting_direction)

                intersection_point = camera_coordinates + t * connecting_direction

                if np.linalg.norm(
                    intersection_point - camera_coordinates
                ) < np.linalg.norm(connecting_line):
                    led["cameras_that_can_see"].remove(camera_object)
                    # Stupid way of checking if the led object (which is a dictionary) is inside the camera_object.leds_visible list
                    index_length = len(camera_object.leds_visible)
                    for index in range(index_length):
                        if camera_object.leds_visible[index] is led:
                            del camera_object.leds_visible[index]
                            break
                    break  # Break statement added because we do not have to test the led anymore

            else:
                continue
