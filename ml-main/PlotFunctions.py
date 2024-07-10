from SpawnFunctions import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_camera_location(axes_object, camera_object, color="black", marker="o"):
    """
    This function plots the camera location with a user specified marker type.
    We need to input the axes object, meaning, the figure and subplots need to be created already.
    Please look at the "Notes" section of the following matplotlib page for more details about the format strings: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    The axes object should belong to a '3d' projection subplot. (ax = fig.add_subplot(projection='3d'))
    This function plots the Cameras location only, if, you want to plot the FoV vectors as well, look at the function "plot_camera_fov()" instead.
    """
    start_point = camera_object.inertial_coordinates.copy()
    if camera_object.type == "wide angle":
        marker = "1"
        color = "cyan"
    elif camera_object.type == "medium range":
        marker = "2"
        color = "green"
    elif camera_object.type == "telephoto":
        marker = "+"
        color = "yellow"
    axes_object.scatter(
        start_point[0],
        start_point[1],
        start_point[2],
        color=color,
        marker=marker,
    )


def plot_event(
    axes_object,
    event_object,
    event_marker="$E$",
    marker_color="yellow",
    marker_alpha=1.0,
):
    # https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers for marker styles
    # https://matplotlib.org/stable/gallery/pyplots/pyplot_text.html for mathtext tips

    coordinates = event_object.inertial_coordinates
    axes_object.scatter(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        color=marker_color,
        marker=event_marker,
        alpha=marker_alpha,
    )


def plot_camera_fov(
    axes_object,
    camera_object,
    fov_line_format="-b",
    fov_line_alpha=1.0,
    plane_line_format="--b",
    plane_line_alpha=1.0,
    start_point_color="black",
    start_point_marker="o",
    end_point_color="green",
    end_point_marker=".",
    marker_alpha=1.0,
):
    """
    This function plots the camera location with a user specified marker type, line segments of the fields of view, and the plane of the view at its maximum range.
    We need to input the axes object, meaning, the figure and subplots need to be created already.
    We can specify the properties of the line segment and the end points, or else the default values (Blue color line, Blue color end points) will be used.
    Please look at the "Notes" section of the following matplotlib page for more details about the format strings: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    The axes object should belong to a '3d' projection subplot. (ax = fig.add_subplot(projection='3d'))
    This function plots the Cameras along with the FoV, if you want to plot just the cameras location, look at the function "plot_camera_location()" instead.
    """

    four_vectors = camera_object.inertial_frame_fov_bounding_vectors.copy()
    pointing = camera_object.inertial_frame_pointing_vector.copy()
    normalized_pointing = pointing / np.linalg.norm(pointing)
    start_point = camera_object.inertial_coordinates.copy()
    max_range = camera_object.max_range

    list_of_end_points = []

    for vector in four_vectors:
        normalized_vector = vector / np.linalg.norm(vector)
        # angle = np.arccos(np.clip(np.dot(pointing, vector), -1.0, 1.0)) # https://stackoverflow.com/a/13849249
        # distance = range / np.cos(angle)
        fov_line_length = max_range / np.dot(normalized_pointing, normalized_vector)
        end_point = start_point + fov_line_length * normalized_vector
        list_of_end_points.append(end_point)
        axes_object.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            [start_point[2], end_point[2]],
            fov_line_format,
            alpha=fov_line_alpha,
        )

        axes_object.scatter(
            end_point[0],
            end_point[1],
            end_point[2],
            color=end_point_color,
            marker=end_point_marker,
            alpha=marker_alpha,
        )

    # Plotting camera location
    axes_object.scatter(
        start_point[0],
        start_point[1],
        start_point[2],
        color=start_point_color,
        marker=start_point_marker,
        alpha=marker_alpha,
    )

    # The plot function takes all x components as a list and them all y components as another list, if you want multiple connected lines.
    axes_object.plot(
        [
            list_of_end_points[0][0],
            list_of_end_points[1][0],
            list_of_end_points[2][0],
            list_of_end_points[3][0],
            list_of_end_points[0][0],
        ],
        [
            list_of_end_points[0][1],
            list_of_end_points[1][1],
            list_of_end_points[2][1],
            list_of_end_points[3][1],
            list_of_end_points[0][1],
        ],
        [
            list_of_end_points[0][2],
            list_of_end_points[1][2],
            list_of_end_points[2][2],
            list_of_end_points[3][2],
            list_of_end_points[0][2],
        ],
        plane_line_format,
        alpha=plane_line_alpha,
    )

    # Plotting the camera up vector
    up_vector_artist = axes_object.quiver(
        start_point[0],
        start_point[1],
        start_point[2],
        camera_object.inertial_frame_up_vector[0],
        camera_object.inertial_frame_up_vector[1],
        camera_object.inertial_frame_up_vector[2],
        length=5,
        arrow_length_ratio=0.2,
        alpha=fov_line_alpha,
    )


def plot_cubesat(
    axes_object,
    cubesat_object,
    edge_line_format="-",
    edge_line_color="white",
    vertex_marker="o",
    vertex_color="white",
):
    """
    This function plots a cubesat vertices, edges and LEDs on the surface.
    We need to input the axes object, meaning, the figure and subplots need to be created already.
    We can specify the properties of the line segment and the end points, or else the default values (Blue color line, Green color end points) will be used.
    Please look at the "Notes" section of the following matplotlib page for more details about the format strings: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    The axes object should belong to a '3d' projection subplot. (ax = fig.add_subplot(projection='3d'))
    """

    vertices = cubesat_object.vertices_list
    edges = cubesat_object.edges_list
    leds = cubesat_object.leds_list

    for vertex in vertices:
        start_point = vertex["inertial_coordinate"]
        for edge in edges:
            if edge["start_point"] == vertex:
                end_point = edge["end_point"]["inertial_coordinate"]
                axes_object.plot(
                    [start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]],
                    linestyle=edge_line_format,
                    color=edge_line_color,
                )

        axes_object.scatter(
            start_point[0],
            start_point[1],
            start_point[2],
            color=vertex_color,
            marker=vertex_marker,
        )

    for led in leds:
        coordinate = led["inertial_coordinate"]
        axes_object.scatter(
            coordinate[0],
            coordinate[1],
            coordinate[2],
            color=led["color"],
            marker="o",
        )

    # Plot camera body frame
    body_origin = cubesat_object.distance_from_inertial_origin
    axes_object.quiver(
        body_origin[0],
        body_origin[1],
        body_origin[2],
        cubesat_object.inertial_frame_x_axis[0],
        cubesat_object.inertial_frame_x_axis[1],
        cubesat_object.inertial_frame_x_axis[2],
        color="red",
        length=6,
        arrow_length_ratio=0.2,
    )

    axes_object.quiver(
        body_origin[0],
        body_origin[1],
        body_origin[2],
        cubesat_object.inertial_frame_y_axis[0],
        cubesat_object.inertial_frame_y_axis[1],
        cubesat_object.inertial_frame_y_axis[2],
        color="blue",
        length=6,
        arrow_length_ratio=0.2,
    )

    axes_object.quiver(
        body_origin[0],
        body_origin[1],
        body_origin[2],
        cubesat_object.inertial_frame_z_axis[0],
        cubesat_object.inertial_frame_z_axis[1],
        cubesat_object.inertial_frame_z_axis[2],
        color="green",
        length=6,
        arrow_length_ratio=0.2,
    )


def plot_camera_pov_3D(
    axes_object,
    camera_object,
    edge_line_format="-",
    edge_line_color="white",
    vertex_marker="o",
    vertex_color="white",
):
    """
    This function plots a cubesat vertices, edges and LEDs on the surface from the camera's PoV.
    It is the same as the previous function, except that the camera will be at the origin, and the relative coordinates of the CubeSat will be plotted.
    We need to input the axes object, meaning, the figure and subplots need to be created already.
    We can specify the properties of the line segment and the end points, or else the default values (Blue color line, Green color end points) will be used.
    Please look at the "Notes" section of the following matplotlib page for more details about the format strings: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    The axes object should belong to a '3d' projection subplot. (ax = fig.add_subplot(projection='3d'))
    """

    orientation_inverse = camera_object.orientation.inv()

    for cubesat_object in camera_object.cubesats_in_fov:
        vertices = cubesat_object.vertices_list
        edges = cubesat_object.edges_list

        for vertex in vertices:
            start_point = vertex["inertial_coordinate"].copy()

            # Following the same methodology as the take_image() method of the camera object.
            camera_to_start_point_inertial_coordinates = (
                start_point - camera_object.inertial_coordinates.copy()
            )
            camera_to_start_point_camera_coordinates = orientation_inverse.apply(
                camera_to_start_point_inertial_coordinates
            )

            for edge in edges:
                if edge["start_point"] == vertex:
                    end_point = edge["end_point"]["inertial_coordinate"].copy()

                    # Following the same methodology as the take_image() method of the camera object.
                    camera_to_end_point_inertial_coordinates = (
                        end_point - camera_object.inertial_coordinates.copy()
                    )
                    camera_to_end_point_camera_coordinates = orientation_inverse.apply(
                        camera_to_end_point_inertial_coordinates
                    )

                    axes_object.plot(
                        [
                            camera_to_start_point_camera_coordinates[0],
                            camera_to_end_point_camera_coordinates[0],
                        ],
                        [
                            camera_to_start_point_camera_coordinates[1],
                            camera_to_end_point_camera_coordinates[1],
                        ],
                        [
                            camera_to_start_point_camera_coordinates[2],
                            camera_to_end_point_camera_coordinates[2],
                        ],
                        linestyle=edge_line_format,
                        color=edge_line_color,
                    )

            axes_object.scatter(
                camera_to_start_point_camera_coordinates[0],
                camera_to_start_point_camera_coordinates[1],
                camera_to_start_point_camera_coordinates[2],
                color=vertex_color,
                marker=vertex_marker,
            )

    for led in camera_object.leds_visible:
        led_coordinate = led["inertial_coordinate"].copy()

        # Following the same methodology as the take_image() method of the camera object.
        camera_led_inertial_coordinates = (
            led_coordinate - camera_object.inertial_coordinates.copy()
        )
        camera_led_camera_coordinates = orientation_inverse.apply(
            camera_led_inertial_coordinates
        )

        axes_object.scatter(
            camera_led_camera_coordinates[0],
            camera_led_camera_coordinates[1],
            camera_led_camera_coordinates[2],
            color=led["color"],
            marker="o",
        )


def plot_camera_pov_2D(
    axes_object,
    camera_object,
    with_plane=False,
    edge_line_format="--",
    edge_line_color="grey",
):
    """
    This function plots the 2D PoV of the specified camera object. It only plots the visible LED lights on CubeSats (if any) with their respective colors.
    The CubeSat faces (the 4 edges that make up the face) of the visible LEDs will be plotted as dashed lines by default (grey color)
    We need to input the axes object, meaning, the figure and subplots need to be created already.
    Please look at the "Notes" section of the following matplotlib page for more details about the format strings: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    """

    for led in camera_object.output:
        axes_object.scatter(
            led[0],
            led[1],
            color=led[2],
            marker="o",
        )

    max_x_image = [0.0]
    min_x_image = [0.0]
    max_y_image = [0.0]
    min_y_image = [0.0]

    for vertex in camera_object.image_vertex_coordinates:
        axes_object.scatter(
            vertex[0],
            vertex[1],
            color="black",
            marker="o",
        )
        if vertex[0] < min_x_image[0]:
            min_x_image[0] = vertex[0]
        if vertex[0] > max_x_image[0]:
            max_x_image[0] = vertex[0]
        if vertex[1] < min_y_image[0]:
            min_y_image[0] = vertex[1]
        if vertex[1] > max_y_image[0]:
            max_y_image[0] = vertex[1]

    axes_object.set_xlim(min_x_image[0], max_x_image[0])
    axes_object.set_ylim(min_y_image[0], max_y_image[0])

    if with_plane:
        orientation_inverse = camera_object.orientation.inv()
        planes_list = []
        for led in camera_object.leds_visible:
            # I want to add the led["plane"] dictionary to faces_list. This is the quick and dirty way that I can think of at this moment.
            flag = []
            for face in planes_list:
                if face is led["plane"]:
                    flag.append(True)
                    break

            if any(flag):
                continue
            else:
                planes_list.append(led["plane"])

        for plane in planes_list:
            plane_points_camera_output = []  # A list of (x,y) tuples

            # plane["points"] is a list of all vertices of the plane
            for point in plane["points"]:
                # Following the same methodology as the take_image() method of the camera object.
                camera_to_plane_point_inertial_coordinate = (
                    point["inertial_coordinate"] - camera_object.inertial_coordinates
                )

                camera_to_plane_point_camera_coordinate = orientation_inverse.apply(
                    camera_to_plane_point_inertial_coordinate
                )

                x_image = -(
                    camera_object.focal_length
                    * camera_to_plane_point_camera_coordinate[0]
                    / camera_to_plane_point_camera_coordinate[2]
                )
                y_image = -(
                    camera_object.focal_length
                    * camera_to_plane_point_camera_coordinate[1]
                    / camera_to_plane_point_camera_coordinate[2]
                )
                # Negative sign because the camera is looking at -z in its body frame

                plane_points_camera_output.append((x_image, y_image))

            # The plot function takes all x components as a list and them all y components as another list, if you want multiple connected lines.
            axes_object.plot(
                [
                    plane_points_camera_output[0][0],
                    plane_points_camera_output[1][0],
                    plane_points_camera_output[2][0],
                    plane_points_camera_output[3][0],
                    plane_points_camera_output[0][0],
                ],
                [
                    plane_points_camera_output[0][1],
                    plane_points_camera_output[1][1],
                    plane_points_camera_output[2][1],
                    plane_points_camera_output[3][1],
                    plane_points_camera_output[0][1],
                ],
                color=edge_line_color,
                linestyle=edge_line_format,
            )
