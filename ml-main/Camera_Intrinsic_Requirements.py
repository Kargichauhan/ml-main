from CamFunctions import *
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colorbar as cb
from matplotlib.gridspec import GridSpec

# Need to specify the path to the ffmpeg.exe in the computer after adding to the PATH to save as .mp4
# ffmpeg is not pre-installed in all computers, so download from: https://ffmpeg.org/download.html#build-windows
# Extract the folder and put in in C drive. Then add to path, verify by typing "ffmpeg" in command prompt.
plt.rcParams["animation.ffmpeg_path"] = r"C:/ffmpeg/bin/ffmpeg.exe"


max_range_vs_real_life_dist = []
real_life_dist_list = []
max_range_vs_focal_length = []
focal_length_list = []
max_range_vs_pixel_size = []
sensor_size_vs_pixel_size = []
sensor_size_list = []
pixel_size_list = []
max_range_vs_real_life_dist_vs_focal_length = []


# aspect_ratio = 62.2 / 48.8  # This is from the FoV
aspect_ratio = 1920 / 1080  # This is for 1080p video
# aspect_ratio = 3280 / 2464  # This is for 4k video

# Doing all calculations in cm then converting in the plotting phase
for real_life_dist in range(1, 100001):
    max_range = real_life_dist * 0.304 / 0.00112
    max_range_vs_real_life_dist.append(max_range)
    real_life_dist_list.append(real_life_dist)

for focal_length in np.arange(0.1, 10, 0.1):
    max_range = 5 * focal_length / 0.00112
    max_range_vs_focal_length.append(max_range)
    focal_length_list.append(focal_length)

for pixel_size in np.arange(0.0001, 0.001, 0.0001):
    sensor_dist = pixel_size * 10
    max_range = 5 * 0.304 / sensor_dist
    max_range_vs_pixel_size.append(max_range)
    sensor_vertical_pixels = np.floor(np.sqrt(8000000 / aspect_ratio))
    sensor_vertical_size = pixel_size * sensor_vertical_pixels
    sensor_horizontal_size = sensor_vertical_size * aspect_ratio
    sensor_size_number = np.sqrt(
        (sensor_vertical_size) ** 2 + (sensor_horizontal_size) ** 2
    )
    sensor_size_list.append(sensor_size_number)
    pixel_size_list.append(pixel_size)

fig1 = plt.figure(figsize=(20, 10))
# plt.subplot_tool(targetfig=fig1) # This is used to adjust the spacing between subplots in real time
plt.subplots_adjust(
    top=0.9, bottom=0.10, left=0.10, right=0.98, hspace=0.25, wspace=0.12
)  # These are the parameters that I found out using the subplots_tool() method

ax1 = fig1.add_subplot(2, 2, 1)
real_life_dist_list = np.array(real_life_dist_list)
real_life_dist_list = real_life_dist_list / 100  # Convert to meters # type: ignore
max_range_vs_real_life_dist = np.array(max_range_vs_real_life_dist)
max_range_vs_real_life_dist = (
    max_range_vs_real_life_dist / 100
)  # Convert to meters # type: ignore
ax1.plot(real_life_dist_list, max_range_vs_real_life_dist)
ax1.set_title("Max. Range vs 3D Distance between 2 points")
ax1.set_xlabel("3D Distance (m)")
ax1.set_ylabel("Max Range (m)")

ax2 = fig1.add_subplot(2, 2, 2)
focal_length_list = np.array(focal_length_list)
focal_length_list = focal_length_list * 10  # Convert to mm # type: ignore
max_range_vs_focal_length = np.array(max_range_vs_focal_length)
max_range_vs_focal_length = (
    max_range_vs_focal_length / 100
)  # Convert to meters # type: ignore
ax2.plot(focal_length_list, max_range_vs_focal_length)
ax2.set_title("Max. Range vs Camera Focal Length")
ax2.set_xlabel("Focal Length (mm)")
ax2.set_ylabel("Max Range (m)")

ax3 = fig1.add_subplot(2, 2, 3)
pixel_size_list = np.array(pixel_size_list)
pixel_size_list = pixel_size_list * 10000  # Convert to microns # type: ignore
max_range_vs_pixel_size = np.array(max_range_vs_pixel_size)
max_range_vs_pixel_size = (
    max_range_vs_pixel_size / 100
)  # Convert to meters # type: ignore
ax3.plot(pixel_size_list, max_range_vs_pixel_size)
ax3.set_title("Max. Range vs Pixel Size")
ax3.set_xlabel("Pixel Size (microns)")
ax3.set_ylabel("Max Range (m)")

ax4 = fig1.add_subplot(2, 2, 4)
ax4.plot(pixel_size_list, sensor_size_list)
ax4.set_title("Sensor Size vs Pixel Size")
ax4.set_xlabel("Pixel Size (cm)")
ax4.set_ylabel("Sensor Size (cm)")


# Doing a 3D plot of the above stuff for better presentation

typical_sensor_sizes = [
    (12.8, 9.3),
    (8.8, 6.6),
    (7.2, 5.4),
    (6.4, 4.8),
    (5.8, 4.3),
    (4.8, 3.6),
    (3.2, 2.4),
]  # [(horizontal,vertical)] in mm

# typical_pixel_sizes = [14.0, 10.0, 7.0, 6.5, 4.6, 3.5, 2.2, 1.7]  # in microns
# Both the above lists were taken from: https://www.vision-doctor.com/en/camera-technology-basics/sensor-and-pixel-sizes.html
typical_pixel_sizes = np.arange(1, 4, 0.025)
typical_pixel_sizes = np.append(typical_pixel_sizes, np.arange(4, 15, 0.25))


# Doing all calculations in cm then converting in the plotting phase
real_life_dist_list = np.arange(1, 1000, 10)  # 1 cm to 10 m
focal_length_list = np.arange(0.1, 10, 0.3)  # 1 mm to 100 mm
# pixel_size = 0.000112  # 1.12 um # from https://www.uctronics.com/arducam-8-mp-sony-visible-light-ir-fixed-focus-camera-module-for-nvidia-jetson-nano.html
pixels_needed_to_resolve = 10


def req_animate_func(i):
    ax2.cla()

    surf_x_data = []
    surf_y_data = []
    surf_z_data = []
    pixel_size = typical_pixel_sizes[i] / 10000

    for real_life_dist in real_life_dist_list:
        for focal_length in focal_length_list:
            max_range = (
                real_life_dist * focal_length / (pixel_size * pixels_needed_to_resolve)
            )

            surf_x_data.append(real_life_dist)
            surf_y_data.append(focal_length)
            surf_z_data.append(max_range)

    surf_x_data = np.array(surf_x_data)
    surf_x_data = surf_x_data / 100  # type: ignore
    surf_y_data = np.array(surf_y_data)
    surf_y_data = surf_y_data * 10  # type: ignore
    surf_z_data = np.array(surf_z_data)
    surf_z_data = surf_z_data / 100  # type: ignore

    # "labelpad" sets the padding in the label text. https://stackoverflow.com/a/6406750
    ax2.set_xlabel("Real World Distance in m", color="k", labelpad=10)
    ax2.set_ylabel("Focal Length in mm", color="k", labelpad=10)
    ax2.set_zlabel("Range in m", color="k", labelpad=15)  # type: ignore

    # Adding a space between labels and ticks. https://stackoverflow.com/a/29524883
    ax2.tick_params(axis="x", which="major", pad=5)
    ax2.tick_params(axis="y", which="major", pad=5)
    ax2.tick_params(axis="z", which="major", pad=7)  # type: ignore

    ax2.set_zlim(0, 90000)  # type: ignore

    # https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf
    surf = ax2.plot_trisurf(surf_x_data, surf_y_data, surf_z_data, cmap="jet", vmin=0, vmax=90000)  # type: ignore

    ax2.view_init(elev=18, azim=-64)  # type: ignore

    ax2.set_title("Pixel size: {:.3f} microns".format(round(typical_pixel_sizes[i], 3)))  # type: ignore

    fig2.colorbar(
        surf,
        shrink=0.1,
        aspect=50,
        cax=cbar_ax,
    )


# https://stackoverflow.com/a/72332268
fig2 = plt.figure(figsize=(20, 10))
gs = GridSpec(1, 2, width_ratios=[0.9, 0.05], wspace=0)
gs.tight_layout(fig2, pad=0)
ax2 = fig2.add_subplot(gs[0], projection="3d")
cbar_ax = fig2.add_subplot(gs[1])

anim = animation.FuncAnimation(
    fig2, req_animate_func, interval=250, frames=len(typical_pixel_sizes), repeat=True
)

# Saving the Animation
# anim.save(r"C://Users/SpaceTREx/Desktop/animate_func.gif", fps=30)

plt.close(fig1)


plt.show()
