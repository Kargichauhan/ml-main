from CubesatFunctions import *
from CamFunctions import *
import pandas as pd
import time


def convert_image_x_y_to_pixels(
    x_image: float,
    y_image: float,
    h_pixel_size: float,
    v_pixel_size: float,
    h_resolution: float,
    v_resolution: float,
):
    """
    This function takes the x_image, y_image from the pinhole camera model,
    which has the center of the frame as the 0,0 point and uses meters as units.
    It converts that into x_pixel, y_pixel which starts at the Top Left corner as the 0,0 point,
    is unitless and and outputs a tuple of (x_pixel, y_pixel).
    h_resolution is number of pixels horizontally, and v_resolution is number of pixels vertically.
    Pixel size should include binning, if that happens. Otherwise, there will be big errors. (Cropping vs binning)
    """

    # First, convert meters to pixels:
    x_pixel_pre_translation = np.round(x_image / h_pixel_size)
    y_pixel_pre_translation = np.round(y_image / v_pixel_size)

    # Now, translate:
    # To move to the top left, we just add! Draw on a piece of paper and check the logic if you have confusions.
    x_pixel = x_pixel_pre_translation + (h_resolution / 2)
    y_pixel = y_pixel_pre_translation + (v_resolution / 2)

    return (x_pixel, y_pixel)


def generate_dataset():
    """
    This function currently does not take any input arguments. It creates an excel sheet with several columns.
    There is one column for each input neuron. 12 LED lights with (x,y,color) each per CubeSat (Extrinsic Parameters),
    one for camera focal length, h_pixel size, v_pixel_size, h_resolution, v_resolution (Intrinsic Parameters) means 41 inputs total.
    4 quaternions and one range as an output means 5 outputs total. This implies there will be 46 columns overall.
    This is for one Camera looking at one CubeSat.
    I am trying to bring all possible quaternions and ranges into this one file so this is probably going to be a massive excel sheet.
    In the current version, the Intrinsic Parameters are not included in this generator, I have hard coded the Intrinsics to match the Arducam due to time crunch.
    """

    data_dict = {
        "LED_1_c": [],
        "LED_1_x": [],
        "LED_1_y": [],
        "LED_2_c": [],
        "LED_2_x": [],
        "LED_2_y": [],
        "LED_3_c": [],
        "LED_3_x": [],
        "LED_3_y": [],
        "LED_4_c": [],
        "LED_4_x": [],
        "LED_4_y": [],
        "LED_5_c": [],
        "LED_5_x": [],
        "LED_5_y": [],
        "LED_6_c": [],
        "LED_6_x": [],
        "LED_6_y": [],
        "LED_7_c": [],
        "LED_7_x": [],
        "LED_7_y": [],
        "LED_8_c": [],
        "LED_8_x": [],
        "LED_8_y": [],
        "LED_9_c": [],
        "LED_9_x": [],
        "LED_9_y": [],
        "LED_10_c": [],
        "LED_10_x": [],
        "LED_10_y": [],
        "LED_11_c": [],
        "LED_11_x": [],
        "LED_11_y": [],
        "LED_12_c": [],
        "LED_12_x": [],
        "LED_12_y": [],
        "q1": [],
        "q2": [],
        "q3": [],
        "q4": [],
        "range": [],
    }
    # Need to add focal length, sensor size (Image size in terms of pixels x pixels actually) and pixel sizes to generalize it for any camera
    # For now, I am taking it for the Arducam UC-626 Rev B: https://www.arducam.com/product/b0196arducam-8mp-1080p-usb-camera-module-1-4-cmos-imx219-mini-uvc-usb2-0-webcam-board-with-1-64ft-0-5m-usb-cable-for-windows-linux-android-and-mac-os/

    sat_1 = make_sat(
        size="1U",
        origin=(0.18, 0, 0),
        orientation=R.identity(),
    )

    cam_1 = make_camera(
        type="wide angle",
        coordinates=(0, 0, 0),
        pointing_vector=(1, 0, 0),
        up_vector=(0, 0, 1),
    )

    list_of_vectors = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1, 1]),
        np.array([1, 1, 1]),
    ]

    # for focal_length in np.arange(0.001, 1.0, 0.005):
    #     # This is the real thing, will give a lot more data, 200 ish focal lengths
    #     # 1 mm focal length to 1 meter in 2 mm steps
    #     # Units are in meters throughout

    #     print(f"Focal length = {focal_length}")

    cam_1.focal_length = 0.00304  # 3.04 mm EFL from Arducam's website

    cam_1.h_resolution = 640.0  # YOLOV5 uses only 640x480
    cam_1.v_resolution = 480.0
    # Use number of pixels, not pixel size in meters or something like that.

    cam_1.h_pixel_size = 5.125 * 0.00000112  # 1.12 microns from Arducam's website
    cam_1.v_pixel_size = 5.125 * 0.00000112
    # 5 ish pixels are binned together in the Arducam
    # 3280 x 3464 is binned to 640 x 480 which is 5.125 x 5.13333... pixels being binned # sigh

    for every_translation in range(66):
        # Distances of 20 cm (from making the sat object) from origin to 20 + 132 = 150 cm from origin in 2 cm steps
        # Units are in meters throughout

        distance = 0.02  # Translate by 2 cm every time in the loop

        print(f"Distance = {sat_1.distance_from_inertial_origin}")

        sat_1.translate(distance=[distance, 0, 0])

        for rotation_vector in list_of_vectors:
            # Go through all vectors in that list

            rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)

            # This is for the 360 x 1 degrees rotations real thing
            rotvec = rotation_vector * 1
            rotation_instance = R.from_rotvec(rotvec=rotvec, degrees=True)

            for i in range(240):
                # All angles from 0 to 358.5 degrees in steps of 1.5
                # We are just doing an range from 0 to 240 as that is the number of times the same rotvec is applied to to the object so that it comes back to the original position.
                # Then the next rotation vector can be used to apply the 720 rotations, and so on.

                sat_1.rotate(rotation_instance=rotation_instance)
                # So, we are rotating by 1.5 degrees 240 times.

                fov_led_test(camera_object=cam_1, cubesat_objects_list=[sat_1])
                los_led_test(camera_object=cam_1)
                cam_1.take_image()

                # Initially, the camera's inertial pointing vector [1,0,0] aligns with the CubeSat's x-axis.
                # Initially, the camera's inertial up vector [0,0,1] aligns with the CubeSat's z-axis. We have set it that way initially, arbitrarily.
                # We are defining that to be identity quaternion between the CubeSat and the camera.
                # So, we calculate the rotation instances between the CubeSat and Camera for future rotations using these two vectors.

                camera_vectors = np.array(
                    [
                        [cam_1.inertial_frame_pointing_vector],
                        [cam_1.inertial_frame_up_vector],
                    ]
                ).reshape(2, 3)
                cubesat_vectors = np.array(
                    [
                        [sat_1.inertial_frame_x_axis],
                        [sat_1.inertial_frame_z_axis],
                    ]
                ).reshape(2, 3)

                (sat_cam_rotation_instance, error) = R.align_vectors(camera_vectors, cubesat_vectors)  # type: ignore
                # (a,b) arguments gives the rotation instance that aligns b to a. This function uses the Kabsch Algorithm.

                sat_cam_quaternion = sat_cam_rotation_instance.as_quat()  # type: ignore

                sat_cam_distance = np.linalg.norm(
                    sat_1.distance_from_inertial_origin - cam_1.inertial_coordinates
                )

                # Start populating the dict one column at a time.

                data_dict["q1"].append(sat_cam_quaternion[0])
                data_dict["q2"].append(sat_cam_quaternion[1])
                data_dict["q3"].append(sat_cam_quaternion[2])
                data_dict["q4"].append(sat_cam_quaternion[3])
                data_dict["range"].append(sat_cam_distance)

                # data_dict["focal_length"].append(cam_1.focal_length)

                # Need to add pixel size and h_ and v_resolutions as well to generalize.

                # The easy stuff is done, now comes the hard part. We will have to trial and error this.
                # For Zero padding, I am going to give 0 as the inputs for x and y, and -1 for color.
                # May need to change this based on the results.

                # camera_object.take_image() outputs a list of tuples -> [(x,y,c),(x,y,c),...].

                # Jaret's code outputs the location in terms of x_pixels, y_pixels, with 0,0 being to left corner.
                # The images are BINNED to 640 x 480 pixels.
                # So we will have to do a conversion from x_image, y_image to x_pixels,y_pixels (Using the pixel size)

                if len(cam_1.leds_visible) > 0:
                    # We add the x,y,color to the dataframe only if there are LEds visible. So we add them one by one.
                    # We add LED 1 if there is atleast 1 LED visible. We add LED 2 if there is atleast 2 LEDs visible, etc.
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[0][0],
                        y_image=cam_1.output[0][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        # Set the LEDs to -1 which go beyond the image size of 640 x 480.
                        # This problem arises due to incorrect FoV of the camera during generation. The previous method used arbitrary FoV.
                        # Now, I have found a better way to do the FoV test using the pixel size and image size. Need to implement that still, will take some time.

                        data_dict["LED_1_x"].append(-cam_1.h_resolution)
                        data_dict["LED_1_y"].append(-cam_1.v_resolution)
                        data_dict["LED_1_c"].append(-1.0)

                    else:
                        # If they don't go beyond the image size of 640 x 480, then we can actually add them.
                        data_dict["LED_1_x"].append(x_pixel)
                        data_dict["LED_1_y"].append(y_pixel)

                        # Arbitrarily saying Red is 0.25, Blue is 0.50, Green is 0.75, Yellow is 1.0. Need to tweak later.
                        if cam_1.output[0][2] == "red":
                            data_dict["LED_1_c"].append(0.25)
                        elif cam_1.output[0][2] == "blue":
                            data_dict["LED_1_c"].append(0.50)
                        elif cam_1.output[0][2] == "green":
                            data_dict["LED_1_c"].append(0.75)
                        elif cam_1.output[0][2] == "yellow":
                            data_dict["LED_1_c"].append(1.0)

                else:
                    # If there are no LEDs visible at all, then we just set it to -1 throughout. (-h_res and -v_res for normalization)
                    data_dict["LED_1_x"].append(-cam_1.h_resolution)
                    data_dict["LED_1_y"].append(-cam_1.v_resolution)
                    data_dict["LED_1_c"].append(-1.0)

                if len(cam_1.leds_visible) > 1:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[1][0],
                        y_image=cam_1.output[1][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_2_x"].append(-cam_1.h_resolution)
                        data_dict["LED_2_y"].append(-cam_1.v_resolution)
                        data_dict["LED_2_c"].append(-1.0)

                    else:
                        data_dict["LED_2_x"].append(x_pixel)
                        data_dict["LED_2_y"].append(y_pixel)

                        if cam_1.output[1][2] == "red":
                            data_dict["LED_2_c"].append(0.25)
                        elif cam_1.output[1][2] == "blue":
                            data_dict["LED_2_c"].append(0.50)
                        elif cam_1.output[1][2] == "green":
                            data_dict["LED_2_c"].append(0.75)
                        elif cam_1.output[1][2] == "yellow":
                            data_dict["LED_2_c"].append(1.0)

                else:
                    data_dict["LED_2_x"].append(-cam_1.h_resolution)
                    data_dict["LED_2_y"].append(-cam_1.v_resolution)
                    data_dict["LED_2_c"].append(-1.0)

                if len(cam_1.leds_visible) > 2:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[2][0],
                        y_image=cam_1.output[2][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_3_x"].append(-cam_1.h_resolution)
                        data_dict["LED_3_y"].append(-cam_1.v_resolution)
                        data_dict["LED_3_c"].append(-1.0)

                    else:
                        data_dict["LED_3_x"].append(x_pixel)
                        data_dict["LED_3_y"].append(y_pixel)

                        if cam_1.output[2][2] == "red":
                            data_dict["LED_3_c"].append(0.25)
                        elif cam_1.output[2][2] == "blue":
                            data_dict["LED_3_c"].append(0.50)
                        elif cam_1.output[2][2] == "green":
                            data_dict["LED_3_c"].append(0.75)
                        elif cam_1.output[2][2] == "yellow":
                            data_dict["LED_3_c"].append(1.0)

                else:
                    data_dict["LED_3_x"].append(-cam_1.h_resolution)
                    data_dict["LED_3_y"].append(-cam_1.v_resolution)
                    data_dict["LED_3_c"].append(-1.0)

                if len(cam_1.leds_visible) > 3:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[3][0],
                        y_image=cam_1.output[3][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_4_x"].append(-cam_1.h_resolution)
                        data_dict["LED_4_y"].append(-cam_1.v_resolution)
                        data_dict["LED_4_c"].append(-1.0)

                    else:
                        data_dict["LED_4_x"].append(x_pixel)
                        data_dict["LED_4_y"].append(y_pixel)

                        if cam_1.output[3][2] == "red":
                            data_dict["LED_4_c"].append(0.25)
                        elif cam_1.output[3][2] == "blue":
                            data_dict["LED_4_c"].append(0.50)
                        elif cam_1.output[3][2] == "green":
                            data_dict["LED_4_c"].append(0.75)
                        elif cam_1.output[3][2] == "yellow":
                            data_dict["LED_4_c"].append(1.0)

                else:
                    data_dict["LED_4_x"].append(-cam_1.h_resolution)
                    data_dict["LED_4_y"].append(-cam_1.v_resolution)
                    data_dict["LED_4_c"].append(-1.0)

                if len(cam_1.leds_visible) > 4:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[4][0],
                        y_image=cam_1.output[4][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_5_x"].append(-cam_1.h_resolution)
                        data_dict["LED_5_y"].append(-cam_1.v_resolution)
                        data_dict["LED_5_c"].append(-1.0)

                    else:
                        data_dict["LED_5_x"].append(x_pixel)
                        data_dict["LED_5_y"].append(y_pixel)

                        if cam_1.output[4][2] == "red":
                            data_dict["LED_5_c"].append(0.25)
                        elif cam_1.output[4][2] == "blue":
                            data_dict["LED_5_c"].append(0.50)
                        elif cam_1.output[4][2] == "green":
                            data_dict["LED_5_c"].append(0.75)
                        elif cam_1.output[4][2] == "yellow":
                            data_dict["LED_5_c"].append(1.0)

                else:
                    data_dict["LED_5_x"].append(-cam_1.h_resolution)
                    data_dict["LED_5_y"].append(-cam_1.v_resolution)
                    data_dict["LED_5_c"].append(-1.0)

                if len(cam_1.leds_visible) > 5:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[5][0],
                        y_image=cam_1.output[5][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_6_x"].append(-cam_1.h_resolution)
                        data_dict["LED_6_y"].append(-cam_1.v_resolution)
                        data_dict["LED_6_c"].append(-1.0)

                    else:
                        data_dict["LED_6_x"].append(x_pixel)
                        data_dict["LED_6_y"].append(y_pixel)

                        if cam_1.output[5][2] == "red":
                            data_dict["LED_6_c"].append(0.25)
                        elif cam_1.output[5][2] == "blue":
                            data_dict["LED_6_c"].append(0.50)
                        elif cam_1.output[5][2] == "green":
                            data_dict["LED_6_c"].append(0.75)
                        elif cam_1.output[5][2] == "yellow":
                            data_dict["LED_6_c"].append(1.0)

                else:
                    data_dict["LED_6_x"].append(-cam_1.h_resolution)
                    data_dict["LED_6_y"].append(-cam_1.v_resolution)
                    data_dict["LED_6_c"].append(-1.0)

                if len(cam_1.leds_visible) > 6:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[6][0],
                        y_image=cam_1.output[6][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_7_x"].append(-cam_1.h_resolution)
                        data_dict["LED_7_y"].append(-cam_1.v_resolution)
                        data_dict["LED_7_c"].append(-1.0)

                    else:
                        data_dict["LED_7_x"].append(x_pixel)
                        data_dict["LED_7_y"].append(y_pixel)

                        if cam_1.output[6][2] == "red":
                            data_dict["LED_7_c"].append(0.25)
                        elif cam_1.output[6][2] == "blue":
                            data_dict["LED_7_c"].append(0.50)
                        elif cam_1.output[6][2] == "green":
                            data_dict["LED_7_c"].append(0.75)
                        elif cam_1.output[6][2] == "yellow":
                            data_dict["LED_7_c"].append(1.0)

                else:
                    data_dict["LED_7_x"].append(-cam_1.h_resolution)
                    data_dict["LED_7_y"].append(-cam_1.v_resolution)
                    data_dict["LED_7_c"].append(-1.0)

                if len(cam_1.leds_visible) > 7:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[7][0],
                        y_image=cam_1.output[7][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_8_x"].append(-cam_1.h_resolution)
                        data_dict["LED_8_y"].append(-cam_1.v_resolution)
                        data_dict["LED_8_c"].append(-1.0)

                    else:
                        data_dict["LED_8_x"].append(x_pixel)
                        data_dict["LED_8_y"].append(y_pixel)

                        if cam_1.output[7][2] == "red":
                            data_dict["LED_8_c"].append(0.25)
                        elif cam_1.output[7][2] == "blue":
                            data_dict["LED_8_c"].append(0.50)
                        elif cam_1.output[7][2] == "green":
                            data_dict["LED_8_c"].append(0.75)
                        elif cam_1.output[7][2] == "yellow":
                            data_dict["LED_8_c"].append(1.0)

                else:
                    data_dict["LED_8_x"].append(-cam_1.h_resolution)
                    data_dict["LED_8_y"].append(-cam_1.v_resolution)
                    data_dict["LED_8_c"].append(-1.0)

                if len(cam_1.leds_visible) > 8:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[8][0],
                        y_image=cam_1.output[8][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_9_x"].append(-cam_1.h_resolution)
                        data_dict["LED_9_y"].append(-cam_1.v_resolution)
                        data_dict["LED_9_c"].append(-1.0)

                    else:
                        data_dict["LED_9_x"].append(x_pixel)
                        data_dict["LED_9_y"].append(y_pixel)

                        if cam_1.output[8][2] == "red":
                            data_dict["LED_9_c"].append(0.25)
                        elif cam_1.output[8][2] == "blue":
                            data_dict["LED_9_c"].append(0.50)
                        elif cam_1.output[8][2] == "green":
                            data_dict["LED_9_c"].append(0.75)
                        elif cam_1.output[8][2] == "yellow":
                            data_dict["LED_9_c"].append(1.0)

                else:
                    data_dict["LED_9_x"].append(-cam_1.h_resolution)
                    data_dict["LED_9_y"].append(-cam_1.v_resolution)
                    data_dict["LED_9_c"].append(-1.0)

                if len(cam_1.leds_visible) > 9:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[9][0],
                        y_image=cam_1.output[9][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_10_x"].append(-cam_1.h_resolution)
                        data_dict["LED_10_y"].append(-cam_1.v_resolution)
                        data_dict["LED_10_c"].append(-1.0)

                    else:
                        data_dict["LED_10_x"].append(x_pixel)
                        data_dict["LED_10_y"].append(y_pixel)

                        if cam_1.output[9][2] == "red":
                            data_dict["LED_10_c"].append(0.25)
                        elif cam_1.output[9][2] == "blue":
                            data_dict["LED_10_c"].append(0.50)
                        elif cam_1.output[9][2] == "green":
                            data_dict["LED_10_c"].append(0.75)
                        elif cam_1.output[9][2] == "yellow":
                            data_dict["LED_10_c"].append(1.0)

                else:
                    data_dict["LED_10_x"].append(-cam_1.h_resolution)
                    data_dict["LED_10_y"].append(-cam_1.v_resolution)
                    data_dict["LED_10_c"].append(-1.0)

                if len(cam_1.leds_visible) > 10:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[10][0],
                        y_image=cam_1.output[10][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_11_x"].append(-cam_1.h_resolution)
                        data_dict["LED_11_y"].append(-cam_1.v_resolution)
                        data_dict["LED_11_c"].append(-1.0)

                    else:
                        data_dict["LED_11_x"].append(x_pixel)
                        data_dict["LED_11_y"].append(y_pixel)

                        if cam_1.output[10][2] == "red":
                            data_dict["LED_11_c"].append(0.25)
                        elif cam_1.output[10][2] == "blue":
                            data_dict["LED_11_c"].append(0.50)
                        elif cam_1.output[10][2] == "green":
                            data_dict["LED_11_c"].append(0.75)
                        elif cam_1.output[10][2] == "yellow":
                            data_dict["LED_11_c"].append(1.0)

                else:
                    data_dict["LED_11_x"].append(-cam_1.h_resolution)
                    data_dict["LED_11_y"].append(-cam_1.v_resolution)
                    data_dict["LED_11_c"].append(-1.0)

                if len(cam_1.leds_visible) > 11:
                    (x_pixel, y_pixel) = convert_image_x_y_to_pixels(
                        x_image=cam_1.output[11][0],
                        y_image=cam_1.output[11][1],
                        h_pixel_size=cam_1.h_pixel_size,
                        v_pixel_size=cam_1.v_pixel_size,
                        h_resolution=cam_1.h_resolution,
                        v_resolution=cam_1.v_resolution,
                    )
                    if (
                        (x_pixel < 0)
                        or (x_pixel > cam_1.h_resolution)
                        or (y_pixel < 0)
                        or (y_pixel > cam_1.v_resolution)
                    ):
                        data_dict["LED_12_x"].append(-cam_1.h_resolution)
                        data_dict["LED_12_y"].append(-cam_1.v_resolution)
                        data_dict["LED_12_c"].append(-1.0)

                    else:
                        data_dict["LED_12_x"].append(x_pixel)
                        data_dict["LED_12_y"].append(y_pixel)

                        if cam_1.output[11][2] == "red":
                            data_dict["LED_12_c"].append(0.25)
                        elif cam_1.output[11][2] == "blue":
                            data_dict["LED_12_c"].append(0.50)
                        elif cam_1.output[11][2] == "green":
                            data_dict["LED_12_c"].append(0.75)
                        elif cam_1.output[11][2] == "yellow":
                            data_dict["LED_12_c"].append(1.0)

                else:
                    data_dict["LED_12_x"].append(-cam_1.h_resolution)
                    data_dict["LED_12_y"].append(-cam_1.v_resolution)
                    data_dict["LED_12_c"].append(-1.0)

    dataset_df = pd.DataFrame(data=data_dict)

    return dataset_df


t1 = time.time()

df_1 = generate_dataset()

t2 = time.time()

print(df_1.head())

print(f"Dataset size = {df_1.shape}")

print(f"Time Taken to generate = {t2-t1} seconds")


try:
    t3 = time.time()
    df_1.to_csv("full_dataset_5_Arducam.csv")
    t4 = time.time()

    print(f"Time Taken to save to csv = {t4-t3} seconds")

except Exception as error_message:
    print(f"csv save error with Error Message: {error_message}")

try:
    t3 = time.time()
    df_1.to_excel(
        "full_dataset_5_Arducam.xlsx"
    )  # If too large a dataset, won't save BEWARE!
    t4 = time.time()

    print(f"Time Taken to save to Excel = {t4-t3} seconds")

except Exception as error_message:
    print(f"csv save error with Error Message: {error_message}")


print("Save complete")
