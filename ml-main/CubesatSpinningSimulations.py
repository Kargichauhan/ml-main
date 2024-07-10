from CubesatFunctions import *
from CamFunctions import *
from PlotFunctions import *
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import matplotlib.ticker as ticker
from functools import partial
import os

# Need to specify the path to the ffmpeg.exe in the computer after adding to the PATH to save as .mp4
# ffmpeg is not pre-installed in all computers, so download from: https://ffmpeg.org/download.html#build-windows
# Extract the folder and put in in C drive. Then add to path, verify by typing "ffmpeg" in command prompt.
plt.rcParams["animation.ffmpeg_path"] = r"C:/ffmpeg/bin/ffmpeg.exe"


def setup_default_axes_spinning(ax1, ax2, ax3, ax4):
    """
    A really long function that sets up all parameters of the GridSpec.
    This has been painstakingly done by trial and error.
    If you want to modify these as the animation progresses, you can modify these parameters separately in the sim_anim_func()
    ax2 is a 3D axes object for the 3D inertial frame.
    ax3 is a 3D axes object for the camera 3D PoV.
    ax1 is a 2D axes object for the camera 2D PoV.
    """

    ax1.cla()
    ax1.clear()
    ax2.cla()
    ax2.clear()
    ax3.cla()
    ax3.clear()
    ax4.cla()
    ax4.clear()

    # Setting facecolor (background) for each axes
    ax1.set_facecolor("black")
    ax2.set_facecolor("black")
    ax3.set_facecolor("black")
    ax4.set_facecolor("black")

    # To actually set the background color to black (for 3D only), the below 3 lines are needed: https://stackoverflow.com/a/51109329
    # .set_pane_color() can also be used: https://stackoverflow.com/a/73125383
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # Changing the axis lines colors (3D): https://stackoverflow.com/q/53549960
    ax2.xaxis.line.set_color("white")
    ax2.yaxis.line.set_color("white")
    ax2.zaxis.line.set_color("white")

    ax3.xaxis.line.set_color("white")
    ax3.yaxis.line.set_color("white")
    ax3.zaxis.line.set_color("white")

    # Changing the axis lines colors (2D): https://stackoverflow.com/q/53549960
    ax1.spines["bottom"].set_color("white")
    ax1.spines["top"].set_color("white")
    ax1.spines["right"].set_color("white")
    ax1.spines["left"].set_color("white")

    # Changing the tick lines colors: https://stackoverflow.com/q/53549960
    ax2.tick_params(axis="x", colors="white", labelcolor="white", pad=-3, which="major")
    ax2.tick_params(axis="y", colors="white", labelcolor="white", pad=-3, which="major")
    ax2.tick_params(axis="z", colors="white", labelcolor="white", pad=0, which="major")

    ax3.tick_params(axis="x", colors="white", labelcolor="white", pad=2, which="major")
    ax3.tick_params(axis="y", colors="white", labelcolor="white", pad=5, which="major")
    # ax3.tick_params(axis="z", colors="white", labelcolor="white", pad=0, which="major")

    ax1.tick_params(axis="x", colors="white", labelcolor="white", pad=2, which="major")
    ax1.tick_params(axis="y", colors="white", labelcolor="white", pad=2, which="major")

    # Changing the label color. "labelpad" sets the padding in the label text. https://stackoverflow.com/a/6406750
    ax2.set_xlabel("x-Distance in cm", color="white", labelpad=-11)
    ax2.set_ylabel("y-Distance in cm", color="white", labelpad=-9)
    ax2.set_zlabel("z-Distance in cm", color="white", labelpad=-5)

    ax3.set_xlabel("x-Distance in cm", color="white", labelpad=2)
    ax3.set_ylabel("y-Distance in cm", color="white", labelpad=8)
    # ax3.set_zlabel("z-Distance in cm", color="white", labelpad=0)

    ax1.set_xlabel("x-Distance in cm", color="white", labelpad=10)
    ax1.set_ylabel("y-Distance in cm", color="white", labelpad=10)

    # Setting label positions:https://stackoverflow.com/a/76829944
    ax1.xaxis.set_label_position("bottom")
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    ax2.xaxis.set_label_position("bottom")
    ax2.yaxis.set_label_position("bottom")
    ax2.zaxis.set_label_position("bottom")

    ax3.xaxis.set_label_position("bottom")
    ax3.yaxis.set_label_position("bottom")
    ax3.zaxis.set_label_position("bottom")

    # The following three lines are used to manipulate the notation in the axes ticks: https://stackoverflow.com/a/25750438
    # Read more: https://matplotlib.org/stable/api/ticker_api.html#module-matplotlib.ticker
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax2.zaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax3.zaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # Setting axes limits. Only 3D necessary. For the 2D camera pov, the limits are set inside the plot function itself.
    ax2.set_xlim(-40, 40)
    ax2.set_ylim(-40, 40)
    ax2.set_zlim(0, 80)

    ax3.set_xlim(-40, 40)  # Pay very close attention to the limits
    ax3.set_ylim(40, -40)  # Especially for the 3D PoV
    ax3.set_zlim(0, 80)  # z-axis should come out of the monitor in the plot

    # Setting views for the 3D plots
    ax2.view_init(elev=30, azim=-60)
    ax3.view_init(elev=-90, azim=-90)  # Setting view on the -X-Y plane (PoV)

    # Setting Zoom Level and Box Aspect Ratio
    # https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect.html
    ax2.set_box_aspect(None, zoom=2.5)
    ax3.set_box_aspect(aspect=(1, 1, 1), zoom=1.5)

    # Setting axes titles
    ax2.set_title("3D Rotations", color="white", pad=80)
    ax3.set_title("Camera PoV in 3D", color="white", pad=15)
    ax1.set_title("2D Image", color="white")


def propagate_camera(
    camera_object,
    timestep,
    desired_inertial_linear_velocity=None,
    desired_inertial_angular_velocity=None,
):
    """This function propagates the given camera object over a timestep.
    The body frame linear and angular velocities need to be pre-defined in the object.
    The inertial frame linear and angular velocities need to be given as arrays in the "Setting up simulation parameters" section
    """
    # Even if angular velocity or linear velocity is given in the body frame, it will have to be propagated in the inertial frame.

    if desired_inertial_linear_velocity:
        camera_object.set_inertial_frame_velocity(
            desired_inertial_linear_velocity=desired_inertial_linear_velocity,
            desired_inertial_angular_velocity=desired_inertial_angular_velocity,
        )

    linear_displacement = camera_object.inertial_frame_linear_velocity * timestep
    camera_object.translate(linear_displacement)

    rotational_displacement_vector = (
        camera_object.inertial_frame_angular_velocity * timestep
    )

    rotation_instance = R.from_rotvec(rotational_displacement_vector)
    # The angular velocity * timestep is the same as "rotvec". https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html

    camera_object.rotate(rotation_instance)

    if desired_inertial_linear_velocity:
        camera_object.set_inertial_frame_velocity(
            desired_inertial_linear_velocity=desired_inertial_linear_velocity,
            desired_inertial_angular_velocity=desired_inertial_angular_velocity,
        )


def propagate_cubesat(
    cubesat_object,
    timestep,
    desired_inertial_linear_velocity=None,
    desired_inertial_angular_velocity=None,
):
    """This function propagates the given cubesat object over a timestep.
    The body frame linear and angular velocities need to be pre-defined in the object.
    The inertial frame linear and angular velocities need to be given as arrays in the "Setting up simulation parameters" section
    """
    # Even if angular velocity or linear velocity is given in the body frame, it will have to be propagated in the inertial frame.

    if desired_inertial_linear_velocity:
        cubesat_object.set_inertial_frame_velocity(
            desired_inertial_linear_velocity=desired_inertial_linear_velocity,
            desired_inertial_angular_velocity=desired_inertial_angular_velocity,
        )

    linear_displacement = cubesat_object.inertial_frame_linear_velocity * timestep
    cubesat_object.translate(linear_displacement)

    rotational_displacement_vector = (
        cubesat_object.inertial_frame_angular_velocity * timestep
    )
    rotation_instance = R.from_rotvec(rotational_displacement_vector)
    # The angular velocity * timestep is the same as "rotvec". https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html

    cubesat_object.rotate(rotation_instance)

    if desired_inertial_linear_velocity:
        cubesat_object.set_inertial_frame_velocity(
            desired_inertial_linear_velocity=desired_inertial_linear_velocity,
            desired_inertial_angular_velocity=desired_inertial_angular_velocity,
        )


def simulate(
    camera_objects_list,
    cubesat_objects_list,
    timestep,
):
    # Propagate camera object

    # If no Inertial Frame Velocities are given, then we simply propagate without setting Inertial Frame Velocities
    if len(desired_camera_inertial_angular_velocities) == 0:
        for i in range(len(camera_objects_list)):
            propagate_camera(camera_objects_list[i], timestep)
    else:
        for i in range(len(camera_objects_list)):
            propagate_camera(
                camera_objects_list[i],
                timestep,
                desired_camera_inertial_linear_velocities[i],
                desired_camera_inertial_angular_velocities[i],
            )

    # Propagate CubeSat object

    # If no Inertial Frame Velocities are given, then we simply propagate without setting Inertial Frame Velocities
    if len(desired_cubesat_inertial_linear_velocities) == 0:
        for j in range(len(cubesat_objects_list)):
            propagate_cubesat(cubesat_objects_list[j], timestep)
    else:  # Else, we set inertial frame velocities at each time step
        for j in range(len(cubesat_objects_list)):
            propagate_cubesat(
                cubesat_objects_list[j],
                timestep,
                desired_cubesat_inertial_linear_velocities[j],
                desired_cubesat_inertial_angular_velocities[j],
            )

    # Do FoV test for all cameras and CubeSats
    for camera_object in camera_objects_list:
        fov_led_test(camera_object, cubesat_objects_list)

    # Do LoS test for all cameras and then take images
    for camera_object in camera_objects_list:
        los_led_test(camera_object)

        camera_object.take_image()


def frame_by_frame_plotter_spinning(
    i,
    timestep,
    time_list,
    camera_objects_list,
    cubesat_objects_list,
):
    """
    The time_list should be of the form np.arange(0,end_time,time_step).
    """

    setup_default_axes_spinning(ax1, ax2, ax3, ax4)
    # If you need to modify the axes parameters from the default, you can do it here.

    simulate(
        camera_objects_list,
        cubesat_objects_list,
        timestep,
    )  # Propagates, does FoV and LoS test, takes image

    # Plot functions for ax1 (Camera frame 2D)
    for camera_object in camera_objects_list:
        plot_camera_pov_2D(
            axes_object=ax1,
            camera_object=camera_object,
            edge_line_format="--",
            with_plane=False,
        )

    # plot functions for ax2 (Inertial frame 3D)
    for camera_object in camera_objects_list:
        plot_camera_fov(
            axes_object=ax2,
            camera_object=camera_object,
            start_point_color="white",
        )

    for cubesat_object in cubesat_objects_list:
        plot_cubesat(
            axes_object=ax2,
            cubesat_object=cubesat_object,
        )

    # Plot functions for ax3 (Camera frame 3D)
    for camera_object in camera_objects_list:
        plot_camera_pov_3D(
            axes_object=ax3,
            camera_object=camera_object,
        )

    # Plot functions for the text
    y_coordinate = [-0.150]

    for cubesat_object in cubesat_objects_list:
        ######### Linear Velocity Text #######
        inertial_frame_linear_velocity = np.around(
            cubesat_object.inertial_frame_linear_velocity, decimals=4
        )
        ax4.text(
            0.15,
            y_coordinate[0],
            f"{cubesat_object.name} linear velocity = {inertial_frame_linear_velocity}",
            color="white",
        )
        y_coordinate[0] = y_coordinate[0] + 0.1

        ########## Displacement Text #######

        displacement = np.around(
            np.linalg.norm(
                cubesat_object.distance_from_inertial_origin
                - camera_objects_list[0].inertial_coordinates
            ),
            decimals=4,
        )
        ax4.text(
            0.15,
            y_coordinate[0],
            f"{cubesat_object.name} displacement = {displacement}",
            color="white",
        )
        y_coordinate[0] = y_coordinate[0] + 0.1

        ##### Angular Velocity Text #######

        inertial_frame_angular_velocity = np.around(
            cubesat_object.inertial_frame_angular_velocity, decimals=4
        )
        ax4.text(
            0.15,
            y_coordinate[0],
            f"{cubesat_object.name} angular velocity = {inertial_frame_angular_velocity}",
            color="white",
        )
        y_coordinate[0] = y_coordinate[0] + 0.1

        ######### Relative Orientation Text ####

        relative_orientation, error = R.align_vectors(  # type: ignore
            cubesat_object.inertial_frame_x_axis.reshape(1, 3),
            camera_objects_list[0].inertial_frame_pointing_vector.reshape(1, 3),
        )
        relative_orientation = np.around(relative_orientation.as_quat(), decimals=4)  # type: ignore

        ax4.text(
            0.15,
            y_coordinate[0],
            f"{cubesat_object.name} relative orientation = {relative_orientation}",
            color="white",
        )
        y_coordinate[0] = y_coordinate[0] + 0.1

    for camera_object in camera_objects_list:
        ######### Linear Velocity Text #######
        inertial_frame_linear_velocity = np.around(
            camera_object.inertial_frame_linear_velocity, decimals=4
        )
        ax4.text(
            0.15,
            y_coordinate[0],
            f"{camera_object.name} linear velocity = {inertial_frame_linear_velocity}",
            color="white",
        )
        y_coordinate[0] = y_coordinate[0] + 0.1

        ##### Angular Velocity Text #######

        inertial_frame_angular_velocity = np.around(
            camera_object.inertial_frame_angular_velocity, decimals=4
        )
        ax4.text(
            0.15,
            y_coordinate[0],
            f"{camera_object.name} angular velocity = {inertial_frame_angular_velocity}",
            color="white",
        )
        y_coordinate[0] = y_coordinate[0] + 0.1

    ####### Timestep Text #########
    y_coordinate[0] = y_coordinate[0] + 0.1
    # https://stackoverflow.com/a/47159480
    ax4.text(0.4, y_coordinate[0], f"Timestep = {time_list[i]:.3f}", color="white")


############################## Setting up the Figure ######################################

fig1 = plt.figure(facecolor="black", figsize=(12, 12))
fig1.canvas.manager.set_window_title("CubeSat spinning simulations")  # type: ignore

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-multicolumn-py
# Have to make this weird GridSpec, because 2x3 and 3x4 sizes lead to overlaps in 3D plots
gs = GridSpec(
    7, 8, figure=fig1, left=0.02, right=0.9, top=0.85, bottom=0.05, hspace=2.2
)
gs.tight_layout(figure=fig1)
ax1 = fig1.add_subplot(gs[3:7, 4:8])  # The 2D image in the right
ax2 = fig1.add_subplot(gs[0:2, 1:3], projection="3d")  # Top Left 3D world space
ax3 = fig1.add_subplot(gs[3:7, 0:4], projection="3d")  # Bottom Left camera PoV in 3D
ax4 = fig1.add_subplot(
    gs[0:3, 4:8]
)  # For adding timestep, velocities, positions of the CubeSats and Camera
fig1.suptitle("CubeSat spinning simulations", color="white")
setup_default_axes_spinning(ax1, ax2, ax3, ax4)

################################### Setting up CubeSats #########################################

(r1, error) = R.align_vectors([[0, 0, -1]], [[1, 1, 1]])  # type: ignore
# Notice the [[]] in the argument of the method. It expects matrices [[]], not column vectors []
# (a,b) aligns b to a
(r2, error) = R.align_vectors([[0, -1, 0]], [[1, 1, 1]])  # type: ignore


sat1 = make_sat(size="1U", origin=(15, 5, 50), orientation=r1)  # type: ignore
sat2 = make_sat(size="1U", origin=(-15, 15, 70), orientation=r1)

sat1.name = "Sat 1"
sat2.name = "Sat 2"

# Only set Body Frame velocities here.
# Inertial Frame Velocities should be set as a separate array at desired_cubesat_inertial_linear_velocities, etc

# sat1.body_frame_angular_velocity = np.array([1, 1, 1])
# sat2.body_frame_angular_velocity = np.array([2, 1, 2])


############################################ Setting up Cameras ########################################
cam1 = make_camera(
    type="wide angle",
    coordinates=[5, 0, 0],
    pointing_vector=[0, 0, 1],
    up_vector=[0, 1, 0],
)

cam1.name = "Camera 1"

cam1.max_range = 100  # I am using all units in cm

# Taken from a vendor site for the ArduCam: https://www.uctronics.com/arducam-8-mp-sony-visible-light-ir-fixed-focus-camera-module-for-nvidia-jetson-nano.html
cam1.focal_length = 0.304  # 3.04 mm

# Taken from a vendor site for the ArduCam: https://www.amazon.com/gp/product/B07YHK63DS/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1
cam1.hfov = np.deg2rad(62.2)  # type: ignore
cam1.vfov = np.deg2rad(48.8)  # type: ignore

cam1.get_fov_bounding_vectors()
cam1.get_image_vertex_coordinates()  # Calling since the cam2 intrinsic parameters have been manually changed earlier

################################# Setting up simulation parameters ################################

# Leave empty if you do not want to set Velocities in the Inertial Frame

desired_camera_inertial_linear_velocities = []
desired_camera_inertial_angular_velocities = []

# desired_camera_inertial_linear_velocities = [[0, 0, 0]]
# desired_camera_inertial_angular_velocities = [[1, 0, 0]]

# Leave empty if you do not want to set Velocities in the Inertial Frame

# desired_cubesat_inertial_linear_velocities = []
# desired_cubesat_inertial_angular_velocities = []

desired_cubesat_inertial_linear_velocities = [
    [0, 0, 0],
    [5, -5, 0],
]

desired_cubesat_inertial_angular_velocities = [
    [0.5, 0.5, 1],
    [0.5, 0.5, 0.5],
]

timestep = 1 / 30  # seconds (for 30 fps, we need 1/30 seconds per frame)
total_duration = 10  # seconds
time_list = np.arange(0, total_duration, timestep)  # frames total over 10 seconds

################################### FuncAnimation Method ################################################

anim = animation.FuncAnimation(
    fig=fig1,
    frames=int(total_duration / timestep),
    interval=timestep,
    func=partial(
        frame_by_frame_plotter_spinning,
        timestep=timestep,
        time_list=time_list,
        camera_objects_list=[cam1],
        cubesat_objects_list=[sat1, sat2],
    ),
)

# Saving the Animation

# https://stackoverflow.com/a/31297262
temp_lst = os.listdir(r"C://Users/Athip\Desktop/CubeSatSpinningSimulations")
number_files = len(temp_lst)

sat_1_ang_vel = np.around(desired_cubesat_inertial_angular_velocities[0], decimals=3)
sat_2_ang_vel = np.around(desired_cubesat_inertial_angular_velocities[1], decimals=3)


file_name = (
    f"{number_files+1}. {sat1.name}{sat_1_ang_vel},{sat2.name}{sat_2_ang_vel}.gif"
)

file_path = r"C://Users/Athip\Desktop/CubeSatSpinningSimulations/" + file_name
# anim.save(filename=file_path, fps=30)


plt.show()
