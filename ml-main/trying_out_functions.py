from CubesatFunctions import *
from CamFunctions import *
from EventFunctions import *
from SpawnFunctions import *
from PlotFunctions import *
import ast
import pandas as pd
from matplotlib.gridspec import GridSpec


# r1 = R.identity()
# # Uses q-convention, HOWEVER, identity quaternion is (1,0,0,0) for some reason! #

# print("Making a 1U Satellite:")
# a = make_sat(size="1U", origin=(30, 30, 30), orientation=r1)

# print("Vertex 3 = {}".format(a.vertex_3))
# print("Vertex 7 = {}".format(a.vertex_7))
# print("Edge 3 to 7 = {}".format(a.edge_3to7))
# print("LED 3 = {}".format(a.led_3))

# r2 = R.from_quat([0, 0, np.sin(np.deg2rad(22.5)), np.cos(np.deg2rad(22.5))])

# print("\nPost Rotation:")

# a.rotate(r2)

# print("Vertex 3 = {}".format(a.vertex_3))
# print("Vertex 7 = {}".format(a.vertex_7))
# print("Edge 3 to 7 = {}".format(a.edge_3to7))
# print("LED 3 = {}".format(a.led_3))


# print("\nPost Translation:")

# a.translate(distance=(2, 2, 2))

# print("\nPost 2nd Rotation:")

# r3 = R.from_quat([0, 0, np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))])

# a.rotate(r3)

# print("\nPost 2nd Translation:")

# a.translate(distance=(2, 2, 2))

# print("Vertex 1 = {}".format(a.vertex_1))
# print("Vertex 3 = {}".format(a.vertex_3))
# print("Vertex 5 = {}".format(a.vertex_5))
# print("Edge 3 to 7 = {}".format(a.edge_3to7))
# print("LED 2 = {}".format(a.led_2))
# print("LED 4 = {}".format(a.led_4))

# print("\nTesting the align_vectors method:")

# (instance, error) = R.align_vectors(  # type: ignore
#     np.array([0, 1, 0]).reshape(1, 3), np.array([1, 0, 0]).reshape(1, 3)
# )
# print("Finding the rotation instance from [1, 0, 0] to [0, 1, 0]")
# print(instance.as_quat())  # type: ignore
# print("Applying a rotation to [1, 0, 0]")
# print(instance.apply(np.array([1, 0, 0])))

# print("\nCamera:")

# b = make_camera(
#     type="wide angle",
#     coordinates=[0, 0, 0],
#     pointing_vector=[1, 0, 0],
#     up_vector=[0, -1, 0],
# )
# print("Body Frame FoV Bounding Vectors = {}".format(b.body_frame_fov_bounding_vectors))
# print(
#     "Inertial Frame FoV Bounding Vectors = {}".format(
#         b.inertial_frame_fov_bounding_vectors
#     )
# )
# print("Body Frame Pointing Vector = {}".format(b.body_frame_pointing_vector))
# print("Inertial Frame Pointing Vector = {}".format(b.inertial_frame_pointing_vector))

# print("\nPost Rotation:")

# b.rotate(r2)
# print("Body Frame FoV Bounding Vectors = {}".format(b.body_frame_fov_bounding_vectors))
# print(
#     "Inertial Frame FoV Bounding Vectors = {}".format(
#         b.inertial_frame_fov_bounding_vectors
#     )
# )
# print("Body Frame Pointing Vector = {}".format(b.body_frame_pointing_vector))
# print("Inertial Frame Pointing Vector = {}".format(b.inertial_frame_pointing_vector))

# print("\nPost Translation:")

# b.translate(distance_array=(3, 5, 7))
# print("Body Frame FoV Bounding Vectors = {}".format(b.body_frame_fov_bounding_vectors))
# print(
#     "Inertial Frame FoV Bounding Vectors = {}".format(
#         b.inertial_frame_fov_bounding_vectors
#     )
# )
# print("Body Frame Pointing Vector = {}".format(b.body_frame_pointing_vector))
# print("Inertial Frame Pointing Vector = {}".format(b.inertial_frame_pointing_vector))
# print("Distance from Inertial Origin = {}".format(b.inertial_coordinates))

# print("\nDoing FoV testing:")

# cam1 = make_camera(
#     type="wide angle",
#     coordinates=[0, 0, 0],
#     pointing_vector=[1, 0, 0],
#     up_vector=[0, 1, 0],
# )
# print(cam1.name)

# event1 = make_event(
#     coordinates=[0.92387953, 0.38268343, 0.38268343],
#     luminosity=0.14,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event2 = make_event(
#     [0.92387953, -0.38268343, 0.38268343],
#     luminosity=0.2,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event3 = make_event(
#     [0.92387953, -0.38268343, -0.38268343],
#     luminosity=0.01,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event4 = make_event(
#     [0.92387953, 0.38268343, -0.38268343],
#     luminosity=0.0,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event5 = make_event(
#     [92.387953, 38, 38.3],
#     luminosity=0.91,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event6 = make_event(
#     [0.92, 0.3, 0.3],
#     luminosity=0.04,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event7 = make_event(
#     [10, 2.2, 0],
#     luminosity=0.1,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event8 = make_event(
#     [1, 0, 0],
#     luminosity=0.4,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event9 = make_event(
#     [10, 0, 0],
#     luminosity=0.9,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )
# event10 = make_event(
#     [10, 2.4, 0],
#     luminosity=0.3,
#     luminosity_rate=0,
#     inertial_frame_linear_velocity=[0, 0, 0],
# )

# print(
#     "Camera 1 Body Frame FoV Bounding Vectors = {}".format(
#         cam1.body_frame_fov_bounding_vectors
#     )
# )
# print(
#     "Camera 1 Inertial Frame FoV Bounding Vectors = {}".format(
#         cam1.inertial_frame_fov_bounding_vectors
#     )
# )
# print(
#     "Camera 1 Body Frame Pointing Vector = {}".format(cam1.body_frame_pointing_vector)
# )
# print(
#     "Camera 1 Inertial Frame Pointing Vector = {}".format(
#         cam1.inertial_frame_pointing_vector
#     )
# )
# print("Camera 1 Distance from Inertial Origin = {}".format(cam1.inertial_coordinates))

# print("FoV Test of Camera 1 and Event 1 = {}".format(fov_test(cam1, event1)))
# print("FoV Test of Camera 1 and Event 2 = {}".format(fov_test(cam1, event2)))
# print("FoV Test of Camera 1 and Event 3 = {}".format(fov_test(cam1, event3)))
# print("FoV Test of Camera 1 and Event 4 = {}".format(fov_test(cam1, event4)))
# print("FoV Test of Camera 1 and Event 5 = {}".format(fov_test(cam1, event5)))
# print("FoV Test of Camera 1 and Event 6 = {}".format(fov_test(cam1, event6)))
# print("FoV Test of Camera 1 and Event 7 = {}".format(fov_test(cam1, event7)))
# print("FoV Test of Camera 1 and Event 8 = {}".format(fov_test(cam1, event8)))
# print("FoV Test of Camera 1 and Event 9 = {}".format(fov_test(cam1, event9)))
# print("FoV Test of Camera 1 and Event 10 = {}".format(fov_test(cam1, event10)))

# print(f"List of Events that can be seen by Cam 1 is {cam1.events_visible}")
# print(f"List of Cameras that can see Event 1 is {event1.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 2 is {event2.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 3 is {event3.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 4 is {event4.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 5 is {event5.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 6 is {event6.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 7 is {event7.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 8 is {event8.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 9 is {event9.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 10 is {event10.cameras_that_can_see}")

# print("\nDoing LoS testing:")

# event3.move_to([1, 0, 0])
# event4.move_to([10, 2, 0])
# event5.move_to([3, 0, 0])
# event6.move_to([3, 1, 0])
# event7.move_to([10, 2.5, 0])
# event8.move_to([3, 0.5, 0])

# los_test(cam1)

# print(f"List of Events that can be seen by Cam 1 is {cam1.events_visible}")
# print(f"List of Cameras that can see Event 1 is {event1.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 2 is {event2.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 3 is {event3.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 4 is {event4.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 5 is {event5.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 6 is {event6.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 7 is {event7.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 8 is {event8.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 9 is {event9.cameras_that_can_see}")
# print(f"List of Cameras that can see Event 10 is {event10.cameras_that_can_see}")


# print("\nDoing luminosity testing:")

# fov_test(cam1, event1)
# fov_test(cam1, event2)
# fov_test(cam1, event3)
# fov_test(cam1, event4)
# fov_test(cam1, event5)
# fov_test(cam1, event6)
# fov_test(cam1, event7)
# fov_test(cam1, event8)
# fov_test(cam1, event9)
# fov_test(cam1, event10)

# los_test(cam1)

# luminosity_test(cam1)

# print(f"List of Events that can be seen by Cam 1 is {cam1.events_visible}")
# print(
#     f"List of Cameras that can see Event 1 is {event1.cameras_that_can_see}, event luminosity of {event1.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 2 is {event2.cameras_that_can_see}, event luminosity of {event2.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 3 is {event3.cameras_that_can_see}, event luminosity of {event3.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 4 is {event4.cameras_that_can_see}, event luminosity of {event4.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 5 is {event5.cameras_that_can_see}, event luminosity of {event5.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 6 is {event6.cameras_that_can_see}, event luminosity of {event6.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 7 is {event7.cameras_that_can_see}, event luminosity of {event7.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 8 is {event8.cameras_that_can_see}, event luminosity of {event8.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 9 is {event9.cameras_that_can_see}, event luminosity of {event9.luminosity}"
# )
# print(
#     f"List of Cameras that can see Event 10 is {event10.cameras_that_can_see}, event luminosity of {event10.luminosity}"
# )

# print("\nDoing range testing:")

# range_test(cam1)

# print(f"List of Events that can be seen by Cam 1 is {cam1.events_visible}")
# print(
#     f"List of Cameras that can see Event 1 is {event1.cameras_that_can_see}, event range of {np.linalg.norm(event1.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 2 is {event2.cameras_that_can_see}, event range of {np.linalg.norm(event2.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 3 is {event3.cameras_that_can_see}, event range of {np.linalg.norm(event3.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 4 is {event4.cameras_that_can_see}, event range of {np.linalg.norm(event4.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 5 is {event5.cameras_that_can_see}, event range of {np.linalg.norm(event5.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 6 is {event6.cameras_that_can_see}, event range of {np.linalg.norm(event6.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 7 is {event7.cameras_that_can_see}, event range of {np.linalg.norm(event7.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 8 is {event8.cameras_that_can_see}, event range of {np.linalg.norm(event8.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 9 is {event9.cameras_that_can_see}, event range of {np.linalg.norm(event9.inertial_coordinates)}"
# )
# print(
#     f"List of Cameras that can see Event 10 is {event10.cameras_that_can_see}, event range of {np.linalg.norm(event10.inertial_coordinates)}"
# )


# print('\nTesting the "ast.literal_eval" method')

# print(ast.literal_eval("[250, 10]"))


# print('\nTesting the "spawn_three_winged_station" function')
# list_of_cameras, dataframe = spawn_three_winged_station(
#     wing_dimension=500,
#     no_of_telephotos=4,
#     telephoto_distribution=20,
#     telephoto_gimbal_limit=10,
#     no_of_mediumranges=12,
#     mediumrange_distribution=75,
#     mediumrange_gimbal_limit=10,
#     no_of_wideangles=30,
#     wideangle_distribution=150,
#     wideangle_gimbal_limit=10,
# )

# print(list_of_cameras[0].name)
# print(list_of_cameras[0].inertial_frame_pointing_vector)
# print(list_of_cameras[70].name)
# print(list_of_cameras[70].inertial_frame_pointing_vector)
# print(dataframe)

# print('\nTesting writing the config files')

# dataframe.to_excel("test_config.xlsx")
# dataframe.to_json("test_config.json")


# print("\nTesting the plot function")

# fig = plt.figure(facecolor="black", figsize=(20, 20))
# ax = fig.add_subplot(projection="3d")
# ax.tick_params(labelcolor="white")

# ax.set_facecolor("0")

# # To actually set the background color to black, the below 3 lines are needed: https://stackoverflow.com/a/51109329
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False

# # Changing the tick lines: https://stackoverflow.com/q/53549960
# ax.tick_params(axis="x", colors="white")
# ax.tick_params(axis="y", colors="white")
# ax.tick_params(axis="z", colors="white")

# # Changing the axis lines: https://stackoverflow.com/q/53549960
# ax.xaxis.line.set_color("white")
# ax.yaxis.line.set_color("white")
# ax.zaxis.line.set_color("white")

# # "labelpad sets the padding in the label text. https://stackoverflow.com/a/6406750"
# ax.set_xlabel("Distance in m", color="white", labelpad=10)
# ax.set_ylabel("Distance in m", color="white", labelpad=10)
# ax.set_zlabel("Distance in m", color="white", labelpad=10)

# # The following three lines are used to manipulate the notation in the axes ticks: https://stackoverflow.com/a/25750438
# # Read more: https://matplotlib.org/stable/api/ticker_api.html#module-matplotlib.ticker
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
# ax.zaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))

# # Cameras Only
# for c in range(len(list_of_cameras)):
#     plot_camera_location(ax, list_of_cameras[c], marker="d", color=("y"))
# ax.view_init(elev=25, azim=35)  # Changes the view angle

# #  Cameras and FoV vectors
# for c in range(len(list_of_cameras)):
#     plot_camera_fov(
#         ax,
#         list_of_cameras[c],
#         fov_line_format="-y",
#         start_point_color="w",
#         plane_line_format="--g",
#     )
# ax.view_init(elev=25, azim=35)  # Changes the view angle

# plt.show()

######################################################################################################################

# print("\nPlotting a CubeSat and a Camera")

# fig1 = plt.figure(facecolor="black", figsize=(20, 20))
# ax1 = fig1.add_subplot(projection="3d")

# fig1.canvas.manager.set_window_title("3D Simulation")  # type: ignore
# fig1.set_size_inches(12, 12)

# ax1.tick_params(labelcolor="white")

# ax1.set_facecolor("0")

# # To actually set the background color to black, the below 3 lines are needed: https://stackoverflow.com/a/51109329
# ax1.xaxis.pane.fill = False
# ax1.yaxis.pane.fill = False
# ax1.zaxis.pane.fill = False

# # Changing the tick lines: https://stackoverflow.com/q/53549960
# ax1.tick_params(axis="x", colors="white")
# ax1.tick_params(axis="y", colors="white")
# ax1.tick_params(axis="z", colors="white")

# # Changing the axis lines: https://stackoverflow.com/q/53549960
# ax1.xaxis.line.set_color("white")
# ax1.yaxis.line.set_color("white")
# ax1.zaxis.line.set_color("white")

# # "labelpad sets the padding in the label text. https://stackoverflow.com/a/6406750"
# ax1.set_xlabel("x-Distance in cm", color="white", labelpad=10)
# ax1.set_ylabel("y-Distance in cm", color="white", labelpad=10)
# ax1.set_zlabel("z-Distance in cm", color="white", labelpad=10)

# # The following three lines are used to manipulate the notation in the axes ticks: https://stackoverflow.com/a/25750438
# # Read more: https://matplotlib.org/stable/api/ticker_api.html#module-matplotlib.ticker
# ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
# ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
# ax1.zaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ax1.set_xlim(-40, 40)
# ax1.set_ylim(-40, 40)
# ax1.set_zlim(0, 80)
# # ax1.set_box_aspect(aspect=(1, 1, 1))

# r4 = R.from_quat(
#     [
#         1 * np.sin(np.deg2rad(-90 / 2)),
#         0,
#         0,
#         np.cos(np.deg2rad(-90 / 2)),
#     ]
# )
# r5 = R.from_quat(
#     [
#         0,
#         0,
#         1 * np.sin(np.deg2rad(90 / 2)),
#         np.cos(np.deg2rad(90 / 2)),
#     ]
# )
# r6 = R.from_quat(
#     [
#         1 * np.sin(np.deg2rad(10 / 2)),
#         1 * np.sin(np.deg2rad(10 / 2)),
#         1 * np.sin(np.deg2rad(10 / 2)),
#         np.cos(np.deg2rad(10 / 2)),
#     ]
# )
# r7 = R.from_quat(
#     [
#         0,
#         1 * np.sin(np.deg2rad(-90 / 2)),
#         0,
#         np.cos(np.deg2rad(-90 / 2)),
#     ]
# )
# (r8, error) = R.align_vectors([[0, 1, 0]], [[1, 1, 1]])  # type: ignore
# # Notice the [[]] in the argument of the method. It expects matrices [[]], not column vectors []
# r9 = R.from_quat(
#     [
#         1 * np.sin(np.deg2rad(90 / 2)),
#         0,
#         0,
#         np.cos(np.deg2rad(90 / 2)),
#     ]
# )
# r10 = R.from_rotvec([np.pi / 6, np.pi / 6, np.pi / 6])

# sat2 = make_sat(size="1U", origin=(30, 30, 30), orientation=r1)
# sat2.move_to([5, 10, 50])

# a.move_to([10, 2, 60])
# a.orientation = R.identity()
# a.rotate(rotation_instance=R.identity())
# # a.rotate(r4)
# a.rotate(r5)
# a.rotate(r5)

# a.get_vertices()
# a.get_edges()
# a.get_leds()
# a.get_planes()

# # sat2.rotate(r5)

# # sat2.rotate(r4)

# a.rotate(r4)
# a.rotate(r4)
# a.rotate(r7)

# sat2.rotate(r8)
# sat2.rotate(r10, rotation_frame="body")
# sat2.rotate(r10, rotation_frame="body")
# sat2.rotate(r10, rotation_frame="body")
# sat2.rotate(r10, rotation_frame="body")


# # sat2.rotate(r6)

# plot_cubesat(
#     axes_object=ax1,
#     cubesat_object=a,
#     edge_line_format="-",
#     edge_line_color="white",
#     vertex_color="white",
# )
# plot_cubesat(
#     axes_object=ax1,
#     cubesat_object=sat2,
#     edge_line_format="-",
#     edge_line_color="white",
#     vertex_color="white",
# )

# cam2 = make_camera(
#     type="wide angle",
#     coordinates=[0, 0, 0],
#     pointing_vector=[0, 0, 1],
#     up_vector=[0, 1, 0],
# )
# cam2.max_range = 100  # I am using all units in cm

# # Taken from a vendor site for the ArduCam: https://www.uctronics.com/arducam-8-mp-sony-visible-light-ir-fixed-focus-camera-module-for-nvidia-jetson-nano.html
# cam2.focal_length = 0.304  # 3.04 mm

# # Taken from a vendor site for the ArduCam: https://www.amazon.com/gp/product/B07YHK63DS/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1
# cam2.hfov = np.deg2rad(62.2)
# cam2.vfov = np.deg2rad(48.8)

# cam2.get_fov_bounding_vectors()

# plot_camera_fov(axes_object=ax1, camera_object=cam2)
# # plt.close()  # This function closes all existing figures. Use this function if you don't want the plots called before to be displayed.
# # plt.show()

# ##########################################################################################################################

# print("\nTesting the plot_camera_pov_3D function")

# fov_led_test(cam2, [a])
# fov_led_test(cam2, [sat2])
# los_led_test(cam2)

# fig3 = plt.figure(facecolor="black", figsize=(20, 20))
# ax3 = fig3.add_subplot(projection="3d")

# fig3.canvas.manager.set_window_title("Camera PoV of the 3D Simulation")  # type: ignore
# fig3.set_size_inches(12, 12)

# ax3.tick_params(labelcolor="white")

# ax3.set_facecolor("0")

# # To actually set the background color to black, the below 3 lines are needed: https://stackoverflow.com/a/51109329
# ax3.xaxis.pane.fill = False
# ax3.yaxis.pane.fill = False
# ax3.zaxis.pane.fill = False

# # Changing the tick lines: https://stackoverflow.com/q/53549960
# ax3.tick_params(axis="x", colors="white")
# ax3.tick_params(axis="y", colors="white")
# ax3.tick_params(axis="z", colors="white")

# # Changing the axis lines: https://stackoverflow.com/q/53549960
# ax3.xaxis.line.set_color("white")
# ax3.yaxis.line.set_color("white")
# ax3.zaxis.line.set_color("white")

# # "labelpad sets the padding in the label text. https://stackoverflow.com/a/6406750"
# ax3.set_xlabel("x-Distance in cm", color="white", labelpad=10)
# ax3.set_ylabel("y-Distance in cm", color="white", labelpad=10)
# ax3.set_zlabel("z-Distance in cm", color="white", labelpad=10)

# # The following three lines are used to manipulate the notation in the axes ticks: https://stackoverflow.com/a/25750438
# # Read more: https://matplotlib.org/stable/api/ticker_api.html#module-matplotlib.ticker
# ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
# ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
# ax3.zaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ax3.set_xlim(-40, 40)
# ax3.set_ylim(40, -40)
# ax3.set_zlim(0, 80)

# plot_camera_pov_3D(ax3, cam2)

# # Setting view on the -X-Y plane (PoV)
# ax3.view_init(elev=-90, azim=-90)

# # Setting Zoom Level and Box Aspect Ratio
# # https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect.html
# ax3.set_box_aspect(aspect=(1, 1, 1), zoom=1)

# # plt.close(fig3)

# ##############################################################################################################################

# print("\nTesting the camera 2D pov plot function")

# fig2 = plt.figure(facecolor="black", figsize=(20, 20))
# fig2.canvas.manager.set_window_title("2D PoV")  # type: ignore

# # Multiply window size by (FoV of the camera by 1.5/10) to make the figure bigger on the display
# fig2.set_size_inches(6.22 * 1.5, 4.88 * 1.5)

# ax2 = fig2.add_subplot()
# ax2.tick_params(labelcolor="white")

# ax2.set_facecolor("0")

# # Changing the tick lines: https://stackoverflow.com/q/53549960
# ax2.tick_params(axis="x", colors="white")
# ax2.tick_params(axis="y", colors="white")

# # # Changing the axis lines: https://stackoverflow.com/q/53549960
# ax2.spines["bottom"].set_color("white")
# ax2.spines["top"].set_color("white")
# ax2.spines["right"].set_color("white")
# ax2.spines["left"].set_color("white")

# # "labelpad sets the padding in the label text. https://stackoverflow.com/a/6406750"
# ax2.set_xlabel("x-Distance in cm", color="white", labelpad=10)
# ax2.set_ylabel("y-Distance in cm", color="white", labelpad=10)

# # The following three lines are used to manipulate the notation in the axes ticks: https://stackoverflow.com/a/25750438
# # Read more: https://matplotlib.org/stable/api/ticker_api.html#module-matplotlib.ticker
# ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))
# ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))

# cam2.get_image_vertex_coordinates()  # Calling since the cam2 intrinsic parameters have been manually changed earlier
# cam2.take_image()
# plot_camera_pov_2D(axes_object=ax2, camera_object=cam2, with_plane=True)
# # plot_camera_pov_2D(axes_object=ax2, camera_object=cam2, with_plane=False)

# # plt.close(fig2)

# plt.show()


# print("Testing if a dictionary can have an object as a key")

# dict1 = {
#     "generic key": ["generic value 1", "generic value 2"],
#     cam1: ["generic value 3", "generic value 4"],
#     event1: ["generic value 5", "generic value 6"],
# }

# dict2 = {
#     cam1: ["generic value 3", "generic value 4"],
#     event1: ["generic value 5", "generic value 6"],
# }

# print(dict1)
# print(dict1.keys())
# print(dict1[cam1])
# print(dict1[cam1][1])
# for key in dict2.keys():
#     print(key.name)

# event1.name = "changed event"

# for key in dict2.keys():
#     print(key.name)

# df1 = pd.DataFrame(dict2)

# print(df1)
# print(type(df1.columns))

test_df = pd.DataFrame()

# Spawning some cameras procedurally
list_of_cameras = []
for i in range(10):
    camera_object_1 = make_camera(
        type="wide angle",
        coordinates=(i, i, i),
        pointing_vector=(np.random.random(), np.random.random(), np.random.random()),
        up_vector=(np.random.random(), np.random.random(), np.random.random()),
    )
    list_of_cameras.append(camera_object_1)

    camera_object_2 = make_camera(
        type="medium range",
        coordinates=(i, i, -i),
        pointing_vector=(np.random.random(), np.random.random(), np.random.random()),
        up_vector=(np.random.random(), np.random.random(), np.random.random()),
    )
    list_of_cameras.append(camera_object_2)

    camera_object_3 = make_camera(
        type="telephoto",
        coordinates=(i, -i, i),
        pointing_vector=(np.random.random(), np.random.random(), np.random.random()),
        up_vector=(np.random.random(), np.random.random(), np.random.random()),
    )
    list_of_cameras.append(camera_object_3)


# Spawning some events procedurally
list_of_events = spawn_random_events(
    no_of_events=100,
    x_y_z_extents_low=(-200, -200, -200),
    x_y_z_extents_high=(200, 200, 200),
)

test_df["Events"] = list_of_events

# Do the FoV, LoS, Range and Luminosity tests
for event_object in list_of_events:
    for camera_object in list_of_cameras:
        fov_test(camera_object=camera_object, event_object=event_object)


for camera_object in list_of_cameras:
    los_test(camera_object=camera_object)
    luminosity_test(camera_object=camera_object)
    range_test(camera_object=camera_object)

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

    test_df[camera_object] = list_of_principal_angles

count = 0
for event_object in list_of_events:
    if event_object.cameras_that_can_see:
        count += 1


# Plotting the cameras and events to visualize
fig1 = plt.figure(facecolor="black", figsize=(24.5, 21))
# Size is 1.75x the number of gridspec

fig1.canvas.manager.set_window_title("Testing Observation Platform Dynamic Simulations")  # type: ignore

gs = GridSpec(
    12, 14, figure=fig1, left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.2
)
gs.tight_layout(figure=fig1)

ax1 = fig1.add_subplot(gs[2:12, 4:14], projection="3d")  # The station plot

ax2 = fig1.add_subplot(gs[1:5, 0:3])  # Events and Cameras text on the top left
ax5 = fig1.add_subplot(gs[1:2, 4:14])  # Station text on the center top

ax3 = fig1.add_subplot(gs[5:8, 0:3])  # Cameras plot on the center left

ax4 = fig1.add_subplot(gs[9:12, 0:3])  # Events plot on the bottom left

fig1.suptitle(
    "Testing Observation Station Dynamic Events Simulations", color="white", fontsize=30
)


def setup_default_axes_test(ax1, ax2, ax3, ax4, ax5):
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
    ax1.set_xlim(-6000, 6000)
    ax1.set_ylim(-6000, 6000)
    ax1.set_zlim(-6000, 6000)
    # for the 2D plots, we can set them in the frame_by_frame_plotter

    # Setting views for the 3D plots
    ax1.view_init(elev=25, azim=35)

    # Setting axes titles
    ax2.set_title("Details", color="white", fontsize=18)
    ax3.set_title("Plot of Cameras", color="white", fontsize=18, pad=20)
    ax4.set_title("Plot of Events", color="white", fontsize=18, pad=60)
    # ax5.set_title("Station Dynamic Simulation", color="white", fontsize=25)
    # ax5 is the title for ax1


setup_default_axes_test(ax1, ax2, ax3, ax4, ax5)

# Plot Cameras and FoV vectors
for camera_object in list_of_cameras:
    if camera_object.events_visible:
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
for event_object in list_of_events:
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


# plt.show()


print(f"Events that can be seen = {count}")

assigning_df = test_df.copy()

# Removing empty rows
assigning_df.dropna(axis=0, thresh=2, inplace=True, ignore_index=True)  # type: ignore
# Thresh is 2 because we have the camera object (Which is not an NaN) in each row as well, so the minimum threshold non-NaN values is 2 instead on 1

# Removing empty columns
assigning_df.dropna(axis=1, thresh=1, inplace=True, ignore_index=True)  # type: ignore

print(assigning_df)

# col_number = assigning_df.iloc[1, 1 : len(assigning_df.columns) + 1].values.argmin()  # type: ignore
# col_number = assigning_df.iloc[1, 1 : len(assigning_df.columns) + 1].idxmin()
# print(col_number)
# print(type(col_number))
# print(col_number[0])
# print(assigning_df.iloc[1])
# print(assigning_df.iloc[1, 1 : len(assigning_df.columns) + 1])
# print(assigning_df.iloc[1, 1 : len(assigning_df.columns) + 1].min())
# min_val = assigning_df.iloc[1, 1 : len(assigning_df.columns) + 1].min()
# for col_no in range(1, len(assigning_df.columns) + 1):
#     if assigning_df.iloc[1, col_no] == min_val:
#         print(col_no)
#         print(min_val)
#         break

# print(assigning_df.iloc[1, col_no])
# print(assigning_df.columns[col_no])


# # https://stackoverflow.com/a/36376046
# print(np.argwhere(assigning_df.notnull().values).tolist())

# argslist = np.argwhere(assigning_df.notnull().values).tolist()

# for item in argslist:
#     if item[1] == 0:
#         argslist.remove(item)

# print(argslist)
# print(assigning_df.iloc[argslist[1][0], argslist[1][1]])

# # https://stackoverflow.com/a/29971188
# principal_angle_counts = assigning_df.count(axis=1)
# print(principal_angle_counts)
# print(principal_angle_counts[principal_angle_counts == 2])
# print(principal_angle_counts[principal_angle_counts == 2].index[0])
# print(principal_angle_counts.idxmin())


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
