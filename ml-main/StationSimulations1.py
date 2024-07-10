from SpawnFunctions import *
from PlotFunctions import *
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import matplotlib.ticker as ticker
from functools import partial
import os


# Need to specify the path to the ffmpeg.exe in the computer after adding to the PATH to save as .mp4
# ffmpeg is not pre-installed in all computers, so download from: https://ffmpeg.org/download.html#build-windows
# Extract the folder and put in in C drive. Then add to path, verify by typing "ffmpeg" in command prompt.
plt.rcParams["animation.ffmpeg_path"] = r"C:/ffmpeg/bin/ffmpeg.exe"

plt.rcParams["grid.color"] = "#ffffff1a"  # https://stackoverflow.com/a/47086538
# RGBA to hex: https://rgbacolorpicker.com/rgba-to-hex


def setup_default_axes_station(ax1, ax2, ax3, ax4, ax5):
    """
    Most of these have been taken from the function "setup_default_axes_spinning()" from CubesatSpinningSimulations.py
    """

    for axes_object in [ax1, ax2, ax3, ax4, ax5]:
        axes_object.cla()
        axes_object.clear()

        # Setting facecolor (background) for each axes
        axes_object.set_facecolor("black")

    # To actually set the background color to black (for 3D only), the below 3 lines are needed: https://stackoverflow.com/a/51109329
    # .set_pane_color() can also be used: https://stackoverflow.com/a/73125383
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # Changing the axis lines (3D): https://stackoverflow.com/q/53549960
    # RGBA to hex: https://rgbacolorpicker.com/rgba-to-hex
    ax1.xaxis.line.set_color("#ffffff1a")
    ax1.yaxis.line.set_color("#ffffff1a")
    ax1.zaxis.line.set_color("#ffffff1a")

    for axes_object in [ax3, ax4]:  # 2D plots
        # Changing the axis lines colors (2D): https://stackoverflow.com/q/53549960
        # RGBA to hex: https://rgbacolorpicker.com/rgba-to-hex
        axes_object.spines["bottom"].set_color("white")
        axes_object.spines["top"].set_color("white")
        axes_object.spines["right"].set_color("white")
        axes_object.spines["left"].set_color("white")

    # Changing the tick lines: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    # RGBA to hex: https://rgbacolorpicker.com/rgba-to-hex
    ax1.tick_params(
        axis="x", color="#ffffff73", labelcolor="#ffffff99", pad=5, which="major"
    )
    ax1.tick_params(
        axis="y", color="#ffffff73", labelcolor="#ffffff99", pad=5, which="major"
    )
    ax1.tick_params(
        axis="z", color="#ffffff73", labelcolor="#ffffff99", pad=15, which="major"
    )

    for axes_object in [ax3, ax4]:  # 2D plots
        axes_object.tick_params(
            axis="x", color="white", labelcolor="white", pad=5, which="major"
        )
        axes_object.tick_params(
            axis="y", color="white", labelcolor="white", pad=5, which="major"
        )

    # Changing the label color. "labelpad" sets the padding in the label text. https://stackoverflow.com/a/6406750
    ax1.set_xlabel("x-Distance in m", color="#ffffffbf", labelpad=15)
    ax1.set_ylabel("y-Distance in m", color="#ffffffbf", labelpad=15)
    ax1.set_zlabel("z-Distance in m", color="#ffffffbf", labelpad=30)

    ax3.set_ylabel("Number of Cameras", color="white", labelpad=12)
    ax4.set_ylabel("Number of Events", color="white", labelpad=12)

    ax3.set_xlabel("Time", color="white", labelpad=12)
    ax4.set_xlabel("Time", color="white", labelpad=12)

    # Setting label positions:https://stackoverflow.com/a/76829944
    ax1.xaxis.set_label_position("bottom")
    ax1.yaxis.set_label_position("bottom")
    ax1.zaxis.set_label_position("bottom")

    ax3.xaxis.set_label_position("bottom")

    ax4.xaxis.set_label_position("top")
    ax4.xaxis.tick_top()

    for axes_object in [ax3, ax4]:  # 2D plots: https://stackoverflow.com/a/76829944
        axes_object.yaxis.set_label_position("right")
        axes_object.yaxis.tick_right()

    # The following three lines are used to manipulate the notation in the axes ticks: https://stackoverflow.com/a/25750438
    # Read more: https://matplotlib.org/stable/api/ticker_api.html#module-matplotlib.ticker
    # Use "%.2e" for scientific notation (for larger distances)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax1.zaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    for axes_object in [ax3, ax4]:  # 2D plots
        axes_object.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        axes_object.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Setting axes limits
    ax1.set_xlim(-150500, 550000)
    ax1.set_ylim(-150500, 550000)
    ax1.set_zlim(-150500, 550000)
    # for the 2D plots, we can set them in the frame_by_frame_plotter

    # Setting views for the 3D plots
    ax1.view_init(elev=25, azim=35)

    # Setting axes titles
    ax2.set_title("Details", color="white", fontsize=18)
    ax3.set_title("Plot of Cameras", color="white", fontsize=18, pad=20)
    ax4.set_title("Plot of Events", color="white", fontsize=18, pad=60)
    # ax5.set_title("Station Dynamic Simulation", color="white", fontsize=25)
    # ax5 is the title for ax1


def propagate_event(events_list, timestep):
    """
    This function propagates the given event object over a timestep.
    The inertial frame linear velocities and luminosity rates need to be pre-defined in the object (At the start of the simulation)
    """
    for event_object in events_list:
        linear_displacement = event_object.inertial_frame_linear_velocity * timestep
        event_object.translate(linear_displacement)
        change_in_luminosity = event_object.luminosity_rate * timestep
        event_object.luminosity = event_object.luminosity + change_in_luminosity


def capture_frame(cameras_list, events_list, camera_type: str):
    """
    The camera type can either be "wide angle", or "telephoto", or "medium range"
    Why is camera type important in this function? Because different types of cameras will have different response times and they will capture frames at different speeds and times.
    This function will run the fov_test, los_test, range_test and luminosity_test on all the specified cameras
    The specified cameras is given because each camera type is unique and will have different frame rates due to different exposure times.
    The list of camera objects can be generated by the "spawn_three_winged_station" function.
    """
    for event_object in events_list:
        for camera_object in cameras_list:
            if camera_object.type == camera_type:
                fov_test(camera_object, event_object)
                los_test(camera_object)
                luminosity_test(camera_object)
                range_test(camera_object)


def make_master_table(list_of_cameras, list_of_events):
    """
    This function generates three DataFrames, one for each type of camera.
    The capture frame function should be called first.
    This function makes a DataFrame of the form:
    {"Events": [List of Events],
    "camera_1":[event_1_principle_rotation_angle, event_2_principle_rotation_angle,.....],
    "camera_2":[event_1_principle_rotation_angle, event_2_principle_rotation_angle,.....], ....}
    Using this DataFrame, we can start assigning the events to the cameras based on algorithms of our choice.
    """
    master_df = pd.DataFrame()

    master_df["Events"] = list_of_events

    # Do the FoV, LoS, Range and Luminosity tests
    for event_object in list_of_events:
        for camera_object in list_of_cameras:
            fov_test(camera_object=camera_object, event_object=event_object)

    for camera_object in list_of_cameras:
        los_test(camera_object=camera_object)
        luminosity_test(camera_object=camera_object)
        range_test(camera_object=camera_object)

    # Finding Principal angles and creating the master dataframe
    for camera_object in list_of_cameras:
        list_of_principal_angles = []
        for event_object in list_of_events:
            if event_object in camera_object.events_visible:
                # (rotation_instance, error) = R.align_vectors(final_vectors, initial_vectors)  # type: ignore
                # (a,b) arguments gives the rotation instance that aligns b to a.

                (rotation_instance, error) = R.align_vectors(  # type: ignore
                    (
                        event_object.inertial_coordinates.reshape(1, 3)
                        - camera_object.inertial_coordinates.reshape(1, 3)
                    ),
                    camera_object.inertial_frame_pointing_vector.reshape(1, 3),
                )

                principal_angle = np.linalg.norm(rotation_instance.as_rotvec())

                list_of_principal_angles.append(principal_angle)
            else:
                list_of_principal_angles.append(float("NaN"))

        master_df[camera_object] = list_of_principal_angles

    return master_df


def assign_events_to_cameras_2(master_table_dataframe):
    """
    This function does the role of the scheduler of the Instruments (The "Instrument Controller").
    It iterates through a list of cameras, The list of camera objects can be generated by the "spawn_three_winged_station" function, and a list of events, which will be spawned during the simulation.
    Then assigns one event to each camera, if possible.
    We need to test different algorithms to schedule the cameras to events more effectively.
    """

    assigning_df = master_table_dataframe.copy()

    # Start while loop:
    while_flag = True

    while while_flag:
        if while_flag == False:
            break

        # Removing empty rows
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
        assigning_df.dropna(axis=0, thresh=2, inplace=True, ignore_index=True)  # type: ignore
        # Thresh is 2 because we have the camera object (Which is not an NaN) in each row as well, so the minimum threshold non-NaN values is 2 instead on 1

        # Removing empty columns
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
        assigning_df.dropna(axis=1, thresh=1, inplace=True, ignore_index=True)  # type: ignore

        if assigning_df.empty:
            while_flag = False
            break

        # First, we find out if there are any rows with only one non-NaN value

        # https://stackoverflow.com/a/29971188
        principal_angle_counts = assigning_df.count(axis=1)
        if any(principal_angle_counts == 2):
            # If this condition is True, atleast one row (Event) of the dataFrame only has one match (Only one camera looking at it)

            row_number = principal_angle_counts[principal_angle_counts == 2].index[0]
            # The above code finds out the row number of the first row with exactly 2 non-NaN values (One for the event object, one for the principal angle)

            # https://stackoverflow.com/a/36376046
            argslist = np.argwhere(assigning_df.notnull().values).tolist()
            # The above code makes a list of [row,column] indices of all non-NaN values in the Dataframe. Then we remove the indices of the event objects
            for item in argslist:
                if item[1] == 0:
                    argslist.remove(item)

            for row_column_pair in argslist:
                if row_column_pair[0] == row_number:
                    col_number = row_column_pair[1]

            # We got the row and column numbers, now we have to assign the event to the object and vice-versa.
            event_object = assigning_df.iloc[row_number, 0]  # type: ignore
            camera_object = assigning_df.columns[col_number]  # type: ignore

            camera_object.event_assigned = event_object  # type: ignore

            if camera_object.type == "telephoto":  # type: ignore
                event_object.camera_assigned["telephoto"] = camera_object
            elif camera_object.type == "medium range":  # type: ignore
                event_object.camera_assigned["medium range"] = camera_object
            elif camera_object.type == "wide angle":  # type: ignore
                event_object.camera_assigned["wide angle"] = camera_object

            # After assigning, we drop the row and column, then move on with the loop.
            assigning_df.drop(labels=row_number, axis="index", inplace=True)
            assigning_df.drop(labels=camera_object, axis="columns", inplace=True)

        else:
            # This means no event has exactly one camera looking at it.
            # Next, we can find the event with the least number of cameras looking at it, and then assign the best camera for it, and then keep repeating it recurrently.

            # https://www.geeksforgeeks.org/get-the-index-of-minimum-value-in-dataframe-column/https://www.geeksforgeeks.org/get-the-index-of-minimum-value-in-dataframe-column/
            row_number = principal_angle_counts.idxmin()
            # The above code finds the index (Row number/event object) with the minimum number of principal angles

            min_val = assigning_df.iloc[row_number, 1 : len(assigning_df.columns) + 1].min()  # type: ignore
            # min_val = assigning_df.iloc[row_number, 1 : len(assigning_df.columns)].min()  # type: ignore
            # The above code gets the minimum principal angle value for the particular row so that we can compare and get the column number

            for col_number in range(1, len(assigning_df.columns) + 1):
                if assigning_df.iloc[row_number, col_number] == min_val:  # type: ignore
                    break

            # We got the row and column numbers, now we have to assign the event to the object and vice-versa.
            event_object = assigning_df.iloc[row_number, 0]  # type: ignore
            camera_object = assigning_df.columns[col_number]  # type: ignore

            camera_object.event_assigned = event_object  # type: ignore

            if camera_object.type == "telephoto":  # type: ignore
                event_object.camera_assigned["telephoto"] = camera_object
            elif camera_object.type == "medium range":  # type: ignore
                event_object.camera_assigned["medium range"] = camera_object
            elif camera_object.type == "wide angle":  # type: ignore
                event_object.camera_assigned["wide angle"] = camera_object

            # After assigning, we drop the row and column, then move on with the loop.
            assigning_df.drop(labels=row_number, axis="index", inplace=True)
            assigning_df.drop(labels=camera_object, axis="columns", inplace=True)


def assign_events_to_cameras_1(cameras_list):
    """
    A makeshift function which just assigns events to cameras based on proximity.
    Nothing fancy, just wanted to do this to test out the tracking function.
    Without some sort of assignment, tracking and camera movement is not possible.
    A better scheduler and event assigner will be designed later on. Probably more than one so that we can compare and contrast using this simulation framework.
    """

    # (rotation_instance, error) = R.align_vectors(final_vectors, initial_vectors)  # type: ignore
    # (a,b) arguments gives the rotation instance that aligns b to a.

    for camera_object in cameras_list:
        # going through all the cameras in the system
        if len(camera_object.events_visible) > 0:
            # only check the cameras that have any "events visible" (passed FoV, LoS, range and luminosity tests)

            (rotation_instance, error) = R.align_vectors(  # type: ignore
                (
                    camera_object.events_visible[0].inertial_coordinates.reshape(1, 3)
                    - camera_object.inertial_coordinates.reshape(1, 3)
                ),
                camera_object.inertial_frame_pointing_vector.reshape(1, 3),
            )

            principal_angle = np.linalg.norm(rotation_instance.as_rotvec())

            camera_object.event_assigned = camera_object.events_visible[0]

            if camera_object.type == "telephoto":
                camera_object.events_visible[0].camera_assigned[
                    "telephoto"
                ] = camera_object
            elif camera_object.type == "medium range":
                camera_object.events_visible[0].camera_assigned[
                    "medium range"
                ] = camera_object
            elif camera_object.type == "wide angle":
                camera_object.events_visible[0].camera_assigned[
                    "wide angle"
                ] = camera_object

            # we do the above steps to get the principal angle to the first event in the list and assign it to the camera
            # the below for loop is then used to fond out which event is closest in principal angle to the camera, and then assign that event to the camera in case of a conflict

            for event_object in camera_object.events_visible:
                (rotation_instance, error) = R.align_vectors(  # type: ignore
                    (
                        event_object.inertial_coordinates.reshape(1, 3)
                        - camera_object.inertial_coordinates.reshape(1, 3)
                    ),
                    camera_object.inertial_frame_pointing_vector.reshape(1, 3),
                )

                angle = np.linalg.norm(rotation_instance.as_rotvec())

                if abs(angle) < abs(principal_angle):
                    # if the angle is less than the current lowest angle, we set that as the lowest angle, and assign that event to the camera

                    principal_angle = angle

                    camera_object.event_assigned = event_object

                    if camera_object.type == "telephoto":
                        event_object.camera_assigned["telephoto"] = camera_object
                    elif camera_object.type == "medium range":
                        event_object.camera_assigned["medium range"] = camera_object
                    elif camera_object.type == "wide angle":
                        event_object.camera_assigned["wide angle"] = camera_object


def propagate_station(cameras_list):
    """
    This function propagates the station, consisting of a list of camera objects over a timestep.
    The list of camera objects can be generated by the "spawn_three_winged_station" function.
    There will not be any linear velocity in the system in this version, only gimbaling of the cameras (Angular Velocity)
    This function is called after the "assign_events_to_cameras" function, so we can assume that the cameras have events assigned to them.
    This function finds out the control input needed to keep the events at the center of its assigned camera's FoV (As close as possible to the pointing vector)
    To do that we use the align_vectors() method to find out the rotation instance needed to align two vectors (Kabsch Algorithm.)
    """

    # (rotation_instance, error) = R.align_vectors(final_vectors, initial_vectors)  # type: ignore
    # (a,b) arguments gives the rotation instance that aligns b to a.

    for camera_object in cameras_list:
        # only have to do this for cameras which have events assigned to them

        if camera_object.event_assigned:
            # we first find out if the gimbal angle required from the initial pointing (set at the station spawn function) is lesser than the gimbal limit.
            # only then can we actually gimbal

            (rotation_instance_1, error) = R.align_vectors(  # type: ignore
                (
                    camera_object.event_assigned.inertial_coordinates.reshape(1, 3)
                    - camera_object.inertial_coordinates.reshape(1, 3)
                ),
                camera_object.inertial_frame_initial_pointing.reshape(1, 3),
            )

            gimbal_required_from_initial = np.linalg.norm(
                rotation_instance_1.as_rotvec()
            )

            if gimbal_required_from_initial < camera_object.gimbal_limit:
                # If we can gimbal, then we find out if the gimbal angle required from the current position is smaller than the gimbal rate limit.
                # If not smaller, then we have to rotate it by the gimbal rate limit, and not more than that.

                (rotation_instance_2, error) = R.align_vectors(  # type: ignore
                    (
                        camera_object.event_assigned.inertial_coordinates.reshape(1, 3)
                        - camera_object.inertial_coordinates.reshape(1, 3)
                    ),
                    camera_object.inertial_frame_pointing_vector.reshape(1, 3),
                )

                gimbal_required_from_current = np.linalg.norm(
                    rotation_instance_2.as_rotvec()
                )

                if gimbal_required_from_current <= camera_object.gimbal_rate_limit:
                    camera_object.rotate(rotation_instance_2)
                else:
                    new_rotation_instance = R.from_rotvec(
                        camera_object.gimbal_rate_limit
                        * rotation_instance_2.as_rotvec()
                        / np.linalg.norm(rotation_instance_2.as_rotvec())
                    )
                    camera_object.rotate(new_rotation_instance)


def simulate_station(
    total_duration, timestep, camera_objects_list, events_list, algorithm="1"
):
    list_of_total_events = []
    list_of_total_cameras = []
    list_of_seen_events = []
    list_of_assigned_events = []
    list_of_occupied_cameras = []
    list_of_assigned_cameras = []

    for i in np.arange(timestep, total_duration + timestep, timestep):
        print(f"Timestep = {i:.3f}")

        propagate_event(events_list=events_list, timestep=timestep)

        capture_frame(
            cameras_list=camera_objects_list,
            events_list=events_list,
            camera_type="wide angle",
        )
        capture_frame(
            cameras_list=camera_objects_list,
            events_list=events_list,
            camera_type="medium range",
        )
        capture_frame(
            cameras_list=camera_objects_list,
            events_list=events_list,
            camera_type="telephoto",
        )

        if algorithm == "1":
            assign_events_to_cameras_1(cameras_list=list_of_cameras)

        if algorithm == "2":
            master_table_dataframe = make_master_table(
                list_of_cameras=camera_objects_list, list_of_events=events_list
            )

            assign_events_to_cameras_2(master_table_dataframe=master_table_dataframe)

        propagate_station(cameras_list=camera_objects_list)

        total_events = len(events_list)
        total_cameras = len(camera_objects_list)
        seen_events = 0
        assigned_events = 0
        occupied_cameras = 0
        assigned_cameras = 0
        assigned_events_list = []
        assigned_cameras_list = []

        for event_object in events_list:
            if len(event_object.cameras_that_can_see) > 0:
                seen_events += 1
            if (
                event_object.camera_assigned["telephoto"]
                or event_object.camera_assigned["medium range"]
                or event_object.camera_assigned["wide angle"]
            ):
                if event_object.camera_assigned["telephoto"]:
                    if (
                        not event_object.camera_assigned["telephoto"]
                        in assigned_cameras_list
                    ):
                        assigned_cameras_list.append(
                            event_object.camera_assigned["telephoto"]
                        )
                if event_object.camera_assigned["medium range"]:
                    if (
                        not event_object.camera_assigned["medium range"]
                        in assigned_cameras_list
                    ):
                        assigned_cameras_list.append(
                            event_object.camera_assigned["medium range"]
                        )
                if event_object.camera_assigned["wide angle"]:
                    if (
                        not event_object.camera_assigned["wide angle"]
                        in assigned_cameras_list
                    ):
                        assigned_cameras_list.append(
                            event_object.camera_assigned["wide angle"]
                        )

        for camera_object in camera_objects_list:
            if len(camera_object.events_visible) > 0:
                occupied_cameras += 1
            if camera_object.event_assigned:
                if not camera_object.event_assigned in assigned_events_list:
                    assigned_events_list.append(camera_object.event_assigned)

        assigned_cameras = len(assigned_events_list)
        assigned_events = len(assigned_events_list)

        list_of_seen_events.append(seen_events)
        list_of_assigned_events.append(assigned_events)
        list_of_occupied_cameras.append(occupied_cameras)
        list_of_assigned_cameras.append(assigned_cameras)
        list_of_total_events.append(total_events)
        list_of_total_cameras.append(total_cameras)

    dictionary = {
        "Time": np.arange(timestep, total_duration + timestep, timestep),
        "Total Events": list_of_total_events,
        "Seen Events": list_of_seen_events,
        "Assigned Events": list_of_assigned_events,
        "Total Cameras": list_of_total_cameras,
        "Occupied Cameras": list_of_occupied_cameras,
        "Assigned Cameras": list_of_assigned_cameras,
    }

    df = pd.DataFrame(dictionary)

    return df


def frame_by_frame_plotter_station(
    i, timestep, time_list, total_duration, camera_objects_list, events_list
):
    """
    The time_list should be of the form np.arange(0,end_time,time_step).
    """

    print(f"Time = {time_list[i]:.3f} of {total_duration:.3f} seconds")

    setup_default_axes_station(ax1, ax2, ax3, ax4, ax5)
    # If you need to modify the axes parameters (such as view_init) from the default that need to vary over time, you can do it here.

    propagate_event(events_list=events_list, timestep=timestep)

    capture_frame(
        cameras_list=camera_objects_list,
        events_list=events_list,
        camera_type="wide angle",
    )
    capture_frame(
        cameras_list=camera_objects_list,
        events_list=events_list,
        camera_type="medium range",
    )
    capture_frame(
        cameras_list=camera_objects_list,
        events_list=events_list,
        camera_type="telephoto",
    )

    assign_events_to_cameras_1(cameras_list=camera_objects_list)

    propagate_station(cameras_list=camera_objects_list)

    # Get data on the events and the cameras for the details and the 2D plots
    # If the 2D plots can be scatter plots, then we can make do without the list that we are appending below.
    # However, if the 2D plots need to be line plots, then we will have to initialize lists outside this function, and keep appending and plotting every timestep.
    total_events = len(events_list)
    total_cameras = len(camera_objects_list)
    seen_events = 0
    assigned_events = 0
    occupied_cameras = 0
    assigned_cameras = 0

    for event_object in events_list:
        if len(event_object.cameras_that_can_see) > 0:
            seen_events += 1
        if (
            event_object.camera_assigned["telephoto"]
            or event_object.camera_assigned["medium range"]
            or event_object.camera_assigned["wide angle"]
        ):
            assigned_events += 1

    for camera_object in camera_objects_list:
        if len(camera_object.events_visible) > 0:
            occupied_cameras += 1
        if camera_object.event_assigned:
            assigned_cameras += 1

    unassigned_cameras = total_cameras - assigned_cameras
    unassigned_events = total_events - assigned_events

    list_of_total_events_plot.append(total_events)
    list_of_total_cameras_plot.append(total_cameras)
    list_of_seen_events_plot.append(seen_events)
    list_of_assigned_events_plot.append(assigned_events)
    list_of_unassigned_events_plot.append(unassigned_events)
    list_of_occupied_cameras_plot.append(occupied_cameras)
    list_of_assigned_cameras_plot.append(assigned_cameras)
    list_of_unassigned_cameras_plot.append(unassigned_cameras)
    list_of_times_plot.append(time_list[i])

    # 3D Plots

    # # Plot Cameras Only
    # for c in range(len(camera_objects_list)):
    #     plot_camera_location(ax1, camera_objects_list[c], marker="d", color=("y"))

    # Plot Cameras and FoV vectors
    for camera_object in camera_objects_list:
        if camera_object.event_assigned:
            plot_camera_fov(
                ax1,
                camera_object=camera_object,
                fov_line_format="-y",
                fov_line_alpha=0.4,
                start_point_color="w",
                plane_line_format="--g",
                plane_line_alpha=0.65,
                marker_alpha=0.6,
            )
        else:
            plot_camera_fov(
                ax1,
                camera_object=camera_object,
                fov_line_format=":r",
                fov_line_alpha=0.15,
                start_point_color="w",
                plane_line_format=":r",
                plane_line_alpha=0.10,
                end_point_color="black",
                marker_alpha=0.1,
            )

    # Plot Events
    for event_object in events_list:
        if len(event_object.cameras_that_can_see) > 0:
            plot_event(
                axes_object=ax1,
                event_object=event_object,
                marker_color="cyan",
                event_marker="$E$",
                marker_alpha=1.0,
            )
        else:
            if event_object.luminosity < 0.0:
                plot_event(
                    axes_object=ax1,
                    event_object=event_object,
                    marker_color="grey",
                    event_marker="$e$",
                    marker_alpha=0.35,
                )
            elif event_object.luminosity < 0.5:
                plot_event(
                    axes_object=ax1,
                    event_object=event_object,
                    marker_color="grey",
                    event_marker="$e$",
                    marker_alpha=0.5,
                )
            else:
                plot_event(
                    axes_object=ax1,
                    event_object=event_object,
                    marker_color="blue",
                    event_marker="$e$",
                    marker_alpha=0.4,
                )

    # 2D Plots

    # Plot Cameras
    ax3.plot(
        list_of_times_plot,
        list_of_total_cameras_plot,
        label="Total Cameras",
        linestyle="-",
        color="blue",
        marker=None,
    )
    ax3.plot(
        list_of_times_plot,
        list_of_occupied_cameras_plot,
        label="Occupied Cameras",
        linestyle="-",
        color="green",
        marker=None,
    )
    ax3.plot(
        list_of_times_plot,
        list_of_assigned_cameras_plot,
        label="Assigned Cameras",
        linestyle="-",
        color="cyan",
        marker=None,
    )
    ax3.plot(
        list_of_times_plot,
        list_of_unassigned_cameras_plot,
        label="Unassigned Cameras",
        linestyle="-",
        color="red",
        marker=None,
    )
    ax3.legend(loc="center right")
    ax3.set_xlim(left=0, right=total_duration)
    ax3.set_ylim(bottom=0, top=total_cameras + 10)

    # Plot Events
    ax4.plot(
        list_of_times_plot,
        list_of_total_events_plot,
        label="Total Events",
        linestyle="-",
        color="blue",
        marker=None,
    )
    ax4.plot(
        list_of_times_plot,
        list_of_seen_events_plot,
        label="Seen Events",
        linestyle="-",
        color="green",
        marker=None,
    )
    ax4.plot(
        list_of_times_plot,
        list_of_assigned_events_plot,
        label="Assigned Events",
        linestyle="-",
        color="cyan",
        marker=None,
    )
    ax4.plot(
        list_of_times_plot,
        list_of_unassigned_events_plot,
        label="Unassigned Events",
        linestyle="-",
        color="red",
        marker=None,
    )
    ax4.legend(loc="center right")
    ax4.set_xlim(left=0, right=total_duration)
    ax4.set_ylim(bottom=0, top=total_events + 10)

    # Text Plots

    # Plot Timestep Text
    ax5.text(
        0.5,
        -0.3,
        f"Time = {time_list[i]:.3f} of {total_duration:.3f} seconds",
        color="white",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=14,
    )
    # Plot the title for the 3D Plot
    ax5.text(
        0.5,
        0.2,
        "Station Dynamic Simulation",
        color="white",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=28,
    )

    # Plot Details Text
    ax2.text(
        0.1,
        0.95,
        f"Number of Events in simulation = {total_events}",
        color="white",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=14,
    )
    ax2.text(
        0.1,
        0.89,
        f"Number of Cameras in station = {total_cameras}",
        color="white",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=14,
    )


######################################## Setting up simulation parameters ################################

timestep = 1 / 30  # seconds (for 30 fps, we need 1/30 seconds per frame)
total_duration = 60  # seconds
time_list = np.arange(timestep, total_duration + timestep, timestep)


# ########################################## Setting up Station #########################################

# list_of_cameras, df = spawn_three_winged_station(
#     wing_dimension=500.0,
#     telephoto_camera={
#         "count": 8,
#         "distribution": 100.0,
#         "max_range": 500000.0,
#         "gimbal_limit": 0.523599,  # 0.523599 radians is 30 degrees (Always use radians as R library uses radians)
#         "gimbal_rate_limit": 0.0174533 * 2 * timestep,  # 0.0174533 radians is 1 degree
#         "min_luminosity": 0.001,
#         "focal_length": 0.025,
#         "fov": (7.5, 7.5),
#     },
#     mediumrange_camera={
#         "count": 8,
#         "distribution": 150.0,
#         "max_range": 150000.0,
#         "gimbal_limit": 0.349066,  # 0.523599 radians is 20 degrees (Always use radians as R library uses radians)
#         "gimbal_rate_limit": 0.0872665 * timestep,  # 0.0872665 radians is 5 degrees
#         "min_luminosity": 0.05,
#         "focal_length": 0.006,
#         "fov": (25, 25),
#     },
#     wideangle_camera={
#         "count": 4,
#         "distribution": 200.0,
#         "max_range": 50000.0,
#         "gimbal_limit": 0.174533,  # 0.523599 radians is 10 degrees (Always use radians as R library uses radians)
#         "gimbal_rate_limit": 0.174533 * timestep,  # 0.174533 radians is 10 degrees
#         "min_luminosity": 0.25,
#         "focal_length": 0.0032,
#         "fov": (90, 90),
#     },
# )


# ############################################ Setting up Events ########################################

# list_of_events = spawn_random_events(
#     no_of_events=150,
#     x_y_z_extents_low=[-90000.0, -90000.0, -90000.0],
#     x_y_z_extents_high=[450000.0, 450000.0, 450000.0],
#     # x_y_z_extents_low=[-50000.0, -50000.0, -50000.0],
#     # x_y_z_extents_high=[200000.0, 200000.0, 200000.0],
#     static_movement=False,
#     static_luminosity=False,
#     velocity_limit=(-7000, 7000),
#     luminosity_limit=(0.0, 1.0),
#     luminosity_rate_limit=(-0.04, 0.04),
# )


# # ############################################ Setting up the Figure ######################################

# fig1 = plt.figure(facecolor="black", figsize=(24.5, 21))
# # Size is 1.75x the number of gridspec

# fig1.canvas.manager.set_window_title("Observation Platform Dynamic Simulations")  # type: ignore


# gs = GridSpec(
#     12, 14, figure=fig1, left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.2
# )
# gs.tight_layout(figure=fig1)

# ax1 = fig1.add_subplot(gs[2:12, 4:14], projection="3d")  # The station plot

# ax2 = fig1.add_subplot(gs[1:5, 0:3])  # Events and Cameras text on the top left
# ax5 = fig1.add_subplot(gs[1:2, 4:14])  # Station text on the center top

# ax3 = fig1.add_subplot(gs[5:8, 0:3])  # Cameras plot on the center left

# ax4 = fig1.add_subplot(gs[9:12, 0:3])  # Events plot on the bottom left

# fig1.suptitle(
#     "Observation Station Dynamic Events Simulations", color="white", fontsize=30
# )

# setup_default_axes_station(ax1, ax2, ax3, ax4, ax5)


# ############################################ FuncAnimation Method #########################################

# # The below list is used only for the plot functions. Please do not use them for calculations or other purposes.
# list_of_total_events_plot = []
# list_of_total_cameras_plot = []
# list_of_seen_events_plot = []
# list_of_assigned_events_plot = []
# list_of_unassigned_events_plot = []
# list_of_occupied_cameras_plot = []
# list_of_assigned_cameras_plot = []
# list_of_unassigned_cameras_plot = []
# list_of_times_plot = []

# anim = animation.FuncAnimation(
#     fig=fig1,
#     frames=int(total_duration / timestep),
#     interval=timestep,
#     func=partial(
#         frame_by_frame_plotter_station,
#         timestep=timestep,
#         time_list=time_list,
#         total_duration=total_duration,
#         camera_objects_list=list_of_cameras,
#         events_list=list_of_events,
#     ),
# )

# Saving the Animation

# https://stackoverflow.com/a/31297262
# temp_lst = os.listdir(r"C://Users/SpaceTREx/Desktop/StationSimulations/")
# number_files = len(temp_lst)

# file_name = f"{number_files+1}. Number of events = {len(list_of_events)}, duration = {round(total_duration,3)}.mp4"

# file_path = r"C://Users/SpaceTREx/Desktop/StationSimulations/" + file_name
# anim.save(filename=file_path, fps=30)

# plt.show()


#################################### Simulation only, No Plotting happening here ######################################
# df1 = simulate_station(
#     total_duration=4,
#     timestep=1 / 30,
#     camera_objects_list=list_of_cameras,
#     events_list=list_of_events,
#     algorithm="1",
# )
# df2 = simulate_station(
#     total_duration=4,
#     timestep=1 / 30,
#     camera_objects_list=list_of_cameras,
#     events_list=list_of_events,
#     algorithm="2",
# )

# with pd.option_context(
#     "display.max_rows",
#     None,
#     "display.max_columns",
#     None,
#     "display.precision",
#     3,
# ):
#     print("Algorithm 1")
#     print(
#         df1[["Seen Events", "Assigned Events", "Occupied Cameras", "Assigned Cameras"]]
#     )
#     print("Algorithm 2")
#     print(
#         df2[["Seen Events", "Assigned Events", "Occupied Cameras", "Assigned Cameras"]]
#     )

#################################### Monte Carlo Simulation ######################################

# Setting up the number of simulations and the number of days per simulation
no_of_simulations = 20
no_of_seconds = 20

# Initializing DataFrames to store the data from multiple simulations
df_seen_events = pd.DataFrame(
    data={"Time": np.arange(timestep, no_of_seconds + timestep, timestep)}
)
df_assigned_events = pd.DataFrame(
    data={"Time": np.arange(timestep, no_of_seconds + timestep, timestep)}
)
df_occupied_cameras = pd.DataFrame(
    data={"Time": np.arange(timestep, no_of_seconds + timestep, timestep)}
)
df_assigned_cameras = pd.DataFrame(
    data={"Time": np.arange(timestep, no_of_seconds + timestep, timestep)}
)
df_total_events = pd.DataFrame(
    data={"Time": np.arange(timestep, no_of_seconds + timestep, timestep)}
)
df_total_cameras = pd.DataFrame(
    data={"Time": np.arange(timestep, no_of_seconds + timestep, timestep)}
)

df_list_1 = [
    df_seen_events,
    df_assigned_events,
    df_total_events,
]
df_list_2 = [
    df_occupied_cameras,
    df_assigned_cameras,
    df_total_cameras,
]

label_list_1 = [
    "Seen Events",
    "Assigned Events",
    "Total Events",
]
label_list_2 = [
    "Occupied Cameras",
    "Assigned Cameras",
    "Total Cameras",
]  # For the plot function

color_list_1 = [
    "black",
    "red",
    "blue",
]
color_list_2 = [
    "green",
    "yellow",
    "cyan",
]  # For the plot function

for simulation in range(1, no_of_simulations + 1):
    print(f"Simulation = {simulation}")
    list_of_cameras, df = spawn_three_winged_station(
        wing_dimension=500.0,
        telephoto_camera={
            "count": 8,
            "distribution": 100.0,
            "max_range": 500000.0,
            "gimbal_limit": 0.523599,  # 0.523599 radians is 30 degrees (Always use radians as R library uses radians)
            "gimbal_rate_limit": 0.0174533
            * 2
            * timestep,  # 0.0174533 radians is 1 degree
            "min_luminosity": 0.001,
            "focal_length": 0.025,
            "fov": (7.5, 7.5),
        },
        mediumrange_camera={
            "count": 8,
            "distribution": 150.0,
            "max_range": 150000.0,
            "gimbal_limit": 0.349066,  # 0.523599 radians is 20 degrees (Always use radians as R library uses radians)
            "gimbal_rate_limit": 0.0872665 * timestep,  # 0.0872665 radians is 5 degrees
            "min_luminosity": 0.05,
            "focal_length": 0.006,
            "fov": (25, 25),
        },
        wideangle_camera={
            "count": 4,
            "distribution": 200.0,
            "max_range": 50000.0,
            "gimbal_limit": 0.174533,  # 0.523599 radians is 10 degrees (Always use radians as R library uses radians)
            "gimbal_rate_limit": 0.174533 * timestep,  # 0.174533 radians is 10 degrees
            "min_luminosity": 0.25,
            "focal_length": 0.0032,
            "fov": (90, 90),
        },
    )

    list_of_events = spawn_random_events(
        no_of_events=150,
        x_y_z_extents_low=[-90000.0, -90000.0, -90000.0],
        x_y_z_extents_high=[450000.0, 450000.0, 450000.0],
        # x_y_z_extents_low=[-50000.0, -50000.0, -50000.0],
        # x_y_z_extents_high=[200000.0, 200000.0, 200000.0],
        static_movement=False,
        static_luminosity=False,
        velocity_limit=(-7000, 7000),
        luminosity_limit=(0.0, 1.0),
        luminosity_rate_limit=(-0.04, 0.04),
    )

    df1 = simulate_station(
        total_duration=no_of_seconds,
        timestep=1 / 30,
        camera_objects_list=list_of_cameras,
        events_list=list_of_events,
        algorithm="2",
    )
    df_seen_events[f"Seen Events {simulation}"] = df1["Seen Events"]
    df_assigned_events[f"Assigned Events {simulation}"] = df1["Assigned Events"]
    df_occupied_cameras[f"Occupied Cameras {simulation}"] = df1["Occupied Cameras"]
    df_assigned_cameras[f"Assigned Cameras {simulation}"] = df1["Assigned Cameras"]
    df_total_events[f"Total Events {simulation}"] = df1["Total Events"]
    df_total_cameras[f"Total Cameras {simulation}"] = df1["Total Cameras"]

fig1 = plt.figure(figsize=(24, 12))
fig1.canvas.manager.set_window_title("Station Running Simulations")  # type: ignore
ax1 = fig1.add_subplot(1, 2, 1)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Events")

ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Cameras")

for i in range(len(df_list_1)):
    df = df_list_1[i]

    df["Mean"] = df.iloc[:, 1:].mean(axis=1, numeric_only=True).round(decimals=3)
    df["STD"] = df.iloc[:, 1:].std(axis=1, numeric_only=True).round(decimals=3)
    # https://stackoverflow.com/a/22149930

    ax1.errorbar(
        df["Time"],
        df["Mean"],
        yerr=df["STD"],
        label=label_list_1[i],
        color=color_list_1[i],
        capsize=5,
    )

for i in range(len(df_list_2)):
    df = df_list_2[i]

    df["Mean"] = df.iloc[:, 1:].mean(axis=1, numeric_only=True).round(decimals=3)
    df["STD"] = df.iloc[:, 1:].std(axis=1, numeric_only=True).round(decimals=3)
    # https://stackoverflow.com/a/22149930

    ax2.errorbar(
        df["Time"],
        df["Mean"],
        yerr=df["STD"],
        label=label_list_2[i],
        color=color_list_2[i],
        capsize=5,
    )


ax1.legend()
ax2.legend()

plt.show()
