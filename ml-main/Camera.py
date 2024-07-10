from scipy.spatial.transform import Rotation as R
import numpy as np


class camera:
    def __init__(self, type: str):
        """
        Generates a camera with a user specified type.
        The type should either be "wide angle" or "medium range" or "telephoto".
        """

        self.type = type
        self.name = f"default {type} camera"

        if type == "wide angle":
            self.fov = (90, 90)
            # Full field of view angle, not the half angle. (horizontal, vertical)
            self.min_luminosity = 0.25
            self.max_range = 5000  # I am coming up with another way of calculating amx range based on resolution in the bottom, which will overwrite this value.
            self.focal_length = 0.00304  # Taken from ArduCam Website: https://www.arducam.com/product/b0196arducam-8mp-1080p-usb-camera-module-1-4-cmos-imx219-mini-uvc-usb2-0-webcam-board-with-1-64ft-0-5m-usb-cable-for-windows-linux-android-and-mac-os/
        elif type == "medium range":
            self.fov = (25, 25)
            # Full field of view angle, not the half angle. (horizontal, vertical)
            self.min_luminosity = 0.05
            self.max_range = 50000  # I am coming up with another way of calculating amx range based on resolution in the bottom, which will overwrite this value.
            self.focal_length = 0.006  # Taken from ArduCam Website: https://docs.arducam.com/Optics-and-Lenses/Introduction/#14-optical-format
        elif type == "telephoto":
            self.fov = (5, 5)
            # Full field of view angle, not the half angle. (horizontal, vertical)
            self.min_luminosity = 0.001
            self.max_range = 500000  # I am coming up with another way of calculating amx range based on resolution in the bottom, which will overwrite this value.
            self.focal_length = 0.025  # Taken from ArduCam Website: https://docs.arducam.com/Optics-and-Lenses/Introduction/#14-optical-format

        self.hfov = np.deg2rad(self.fov[0])
        self.vfov = np.deg2rad(self.fov[1])

        self.h_resolution = 1920.0  # No. of pixels in the image, not the sensor
        self.v_resolution = 1080.0  # No. of pixels in the image, not sensor

        self.h_pixel_size = 0.00000112  # 1.12 microns from Arducam's website
        self.v_pixel_size = 0.00000112
        # Pixel size should include binning, if that happens
        # YOLOV5 apparently uses 640x480 by binning 5 ish pixels together, so we have to multiply the above value by 5.

        # Need to overwrite the max range value of the camera based on the above resolution, focal length, and pixel size.
        # Need to do this later. Will improve the calculations and the logic a lot.

        self.orientation = R.identity()
        # We start at identity attitude #

        self.inertial_coordinates = np.array([0.0, 0.0, 0.0])
        # We start by placing the camera at the Origin #

        self.body_frame_pointing_vector = np.array([0.0, 0.0, -1.0])
        # The camera points in the -z direction in the body frame. Only then will the 2D image match the 3D PoV. #
        self.inertial_frame_pointing_vector = self.body_frame_pointing_vector.copy()

        self.inertial_frame_initial_pointing = np.array([0.0, 0.0, 0.0])
        # Use the above attribute for the space station. It will be one of the axes (x,y,z) based on which wing of the station contains the camera #
        # This vector will be used to find out what is the current gimbal state of the camera #

        self.gimbal_limit = 0.0  # Use this attribute in the space station. Can be a tuple if you want two different angles in two different axes.
        self.gimbal_rate_limit = 0.0  # Use this attribute in the space station. Can be a tuple if you want two different angles in two different axes.

        self.body_frame_up_vector = np.array([0.0, 1.0, 0.0])
        self.inertial_frame_up_vector = self.body_frame_up_vector.copy()

        self.body_frame_linear_velocity = np.array([0.0, 0.0, 0.0])
        self.inertial_frame_linear_velocity = self.body_frame_linear_velocity.copy()

        self.body_frame_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.inertial_frame_angular_velocity = self.body_frame_angular_velocity.copy()

        self.body_frame_fov_bounding_vectors = np.array(
            [
                [
                    np.cos(self.vfov / 2) * np.sin(self.hfov / 2),
                    np.sin(self.vfov / 2),
                    -np.cos(self.vfov / 2) * np.cos(self.hfov / 2),
                ],
                [
                    np.cos(self.vfov / 2) * np.sin(self.hfov / 2),
                    -np.sin(self.vfov / 2),
                    -np.cos(self.vfov / 2) * np.cos(self.hfov / 2),
                ],
                [
                    -np.cos(self.vfov / 2) * np.sin(self.hfov / 2),
                    -np.sin(self.vfov / 2),
                    -np.cos(self.vfov / 2) * np.cos(self.hfov / 2),
                ],
                [
                    -np.cos(self.vfov / 2) * np.sin(self.hfov / 2),
                    np.sin(self.vfov / 2),
                    -np.cos(self.vfov / 2) * np.cos(self.hfov / 2),
                ],
            ]
        )
        self.inertial_frame_fov_bounding_vectors = self.body_frame_fov_bounding_vectors

        # Append or remove visible events that satisfy the FoV, LoS, Luminosity and distance tests
        self.events_visible = []  # List of event objects
        self.event_assigned = None
        # Event object assigned by "assign_events_to_cameras" function

        self.cubesats_in_fov = []
        # list of all cubesats within its FoV, not necessarily LoS

        self.leds_visible = []  # List of LEDs (dictionaries)
        self.output = []  # List of tuples (x,y,color)
        self.image_vertex_coordinates = []  # List of 4 (x,y) tuples
        self.get_image_vertex_coordinates()  # Calling this function once so that we will have the image vertices already stored.

    def rotate(self, rotation_instance, rotation_frame="inertial"):
        """
        This method rotates the camera by "rotation_instance".
        Make sure to specify the frame of rotations ("inertial" by default or "body")
        It also stores the cumulative rotations (Current Attitude)
        The "rotation_instance" input argument should be a "Rotation" instance from scipy.spatial.transform
        Make sure to note that the rotation is a Inertial-Frame rotation if a frame parameter is not passed.
        The "rotation_instance" uses q-convention if quaternions are used [vector*sin(), angle*cos()], beware.
        """

        # So the methodology being followed is this:
        # We record the attitude of the body by composing the previous attitude with the rotation_instance to be applied now.
        # Then, we apply the rotation by the new attitude.
        # We have to perform the rotation about the inertial origin only, as the rotation method works that way
        # Uses q-convention if quaternions [vector*sin(), angle*cos()], Beware!!!

        if rotation_frame == "body":
            rotvec_inertial_frame = rotation_instance.as_rotvec()
            rotvec_body_frame = self.orientation.apply(rotvec_inertial_frame)
            # We now have converted the rotation vector from the body frame to the inertial frame.
            # The scipy rotation library only allows rotations about the inertial frame.
            rotation_instance_body_frame = R.from_rotvec(rotvec_body_frame)  # type: ignore
            self.orientation = rotation_instance_body_frame * self.orientation

            self.inertial_frame_pointing_vector = self.orientation.apply(
                self.body_frame_pointing_vector
            )

            self.inertial_frame_up_vector = self.orientation.apply(
                self.body_frame_up_vector
            )

            self.inertial_frame_fov_bounding_vectors = self.orientation.apply(
                self.body_frame_fov_bounding_vectors
            )

            self.inertial_frame_linear_velocity = self.orientation.apply(
                self.body_frame_linear_velocity
            )
            self.inertial_frame_angular_velocity = self.orientation.apply(
                self.body_frame_angular_velocity
            )

        else:
            # This rotation composition takes the previous rotation state, and composes another rotation to it to keep track #
            self.orientation = rotation_instance * self.orientation
            # self.orientation is the current rotation instance of the camera body wrt the inertial frame.

            # Perform rotation about inertial origin #
            self.inertial_frame_pointing_vector = self.orientation.apply(
                self.body_frame_pointing_vector
            )

            self.inertial_frame_up_vector = self.orientation.apply(
                self.body_frame_up_vector
            )

            self.inertial_frame_fov_bounding_vectors = self.orientation.apply(
                self.body_frame_fov_bounding_vectors
            )

            self.inertial_frame_linear_velocity = self.orientation.apply(
                self.body_frame_linear_velocity
            )
            self.inertial_frame_angular_velocity = self.orientation.apply(
                self.body_frame_angular_velocity
            )

    def translate(self, distance_array):
        """
        This method moves the camera by "distance".
        It also stores the cumulative distance moved by the camera in all three coordinates.
        The "distance" input argument should be a 3-array like.
        """

        distance_array = np.array(distance_array).reshape((3,))

        self.inertial_coordinates = (
            self.inertial_coordinates + distance_array
        )  # Adding new translation to the previous distance to keep track #

    def move_to(self, inertial_coordinates):
        """
        This method just moves the camera object to a particular coordinate instead of translating it.
        """
        self.inertial_coordinates = np.array(inertial_coordinates).reshape(
            3,
        )

    def set_inertial_frame_velocity(
        self, desired_inertial_linear_velocity, desired_inertial_angular_velocity
    ):
        """
        This is how you set the velocity of the body in the inertial frame.
        If you simply try to change the inertial_frame_velocity parameter, it won't work.
        The inertial_frame_velocities are parameters that get recomputed based on the body_frame_velocities constantly.
        This function takes the desired inertial_frame_velocities, converts them to body frame velocities anf then sets the body_frame_velocities accordingly.
        IMPORTANT:
        If both translational and rotational velocities are being used in a simulation, then the propagator should set the velocities at each and every time step, especially the linear velocity.
        "Rotational velocity" about the inertial frame does not mean "Revolution" of the body about the Inertial Frame. It merely means the rotation of the body, but about the inertial frame axes.
        """
        orientation_inverse = self.orientation.inv()
        # self.orientation takes body frame and converts it into inertial frame.
        # So orientation_inverse should take inertial frame vectors and convert to body frame.

        self.body_frame_linear_velocity = orientation_inverse.apply(
            np.array(desired_inertial_linear_velocity)
        )
        self.body_frame_angular_velocity = orientation_inverse.apply(
            np.array(desired_inertial_angular_velocity)
        )

        self.rotate(R.identity())

    def get_fov_bounding_vectors(self):
        """This method is used when the FoV of the camera is to be changed.
        The user can modify the hfov and vfov parameters separately in the code and then they must call this method to modify the FoV vectors accordingly.
        If this method is not called after modifying the FoV parameters, then there will be errors, as the FoV values do not correspond to the FoV vectors.
        Only change the self.fov tuple (the full field of view angles), do not change the hfov or the vfov parameters
        """

        self.hfov = np.deg2rad(self.fov[0])
        self.vfov = np.deg2rad(self.fov[1])

        self.body_frame_fov_bounding_vectors = np.array(
            [
                [
                    np.cos(self.vfov / 2) * np.sin(self.hfov / 2),
                    np.sin(self.vfov / 2),
                    -np.cos(self.vfov / 2) * np.cos(self.hfov / 2),
                ],
                [
                    np.cos(self.vfov / 2) * np.sin(self.hfov / 2),
                    -np.sin(self.vfov / 2),
                    -np.cos(self.vfov / 2) * np.cos(self.hfov / 2),
                ],
                [
                    -np.cos(self.vfov / 2) * np.sin(self.hfov / 2),
                    -np.sin(self.vfov / 2),
                    -np.cos(self.vfov / 2) * np.cos(self.hfov / 2),
                ],
                [
                    -np.cos(self.vfov / 2) * np.sin(self.hfov / 2),
                    np.sin(self.vfov / 2),
                    -np.cos(self.vfov / 2) * np.cos(self.hfov / 2),
                ],
            ]
        )

        self.rotate(R.identity())

    def point_to(self, pointing_vector, up_vector):
        """
        This method is used to make the camera point to a particular direction given by "pointing_vector".
        The pointing vector if the camera in the inertial frame is made to align with the specified vector.
        """
        pointing_vector = np.array(pointing_vector) / np.linalg.norm(pointing_vector)
        heading_vector = np.array(up_vector) / np.linalg.norm(up_vector)

        if abs(np.dot(np.array(pointing_vector), np.array(up_vector))) > 0.000001:
            # I am not putting !=0 because there may be round off or approximation errors
            print(
                "The desired pointing and heading (up) vectors are not orthogonal, only the pointing vector will be used, the heading vector will be arbitrary."
            )
            final_pointing = np.array(pointing_vector).reshape(1, 3)
            (rotation_instance, error) = R.align_vectors(final_pointing, self.inertial_frame_pointing_vector.reshape(1, 3))  # type: ignore
            self.rotate(rotation_instance)

        else:
            initial_vectors = np.array(
                [[self.inertial_frame_pointing_vector], [self.inertial_frame_up_vector]]
            ).reshape(2, 3)
            final_vectors = np.array([[pointing_vector], [heading_vector]]).reshape(
                2, 3
            )

            (rotation_instance, error) = R.align_vectors(final_vectors, initial_vectors)  # type: ignore
            # (a,b) arguments gives the rotation instance that aligns b to a. This function uses the Kabsch Algorithm.

            self.rotate(rotation_instance)

    def take_image(self):
        """
        This method produces an 2D array of points and colors corresponding to what the camera sees.
        The output is an array of tuples of ((x,y) coordinates, and corresponding colors) -> [(x1,y1,color1),(x2,y2,color2),(x3,y3,color3),...]
        It converts 3D coordinates of the points it can see into 2D by applying a perspective projection based on the pinhole camera model.
        The formula for the projection is here: https://cseweb.ucsd.edu/classes/fa12/cse252A-a/lec4.pdf
        """

        # The camera coordinate system has a rotation instance of "self.orientation" wrt. the inertial frame.
        # To convert the LED position vector from the inertial frame to the camera frame, we rotate the vector in the inverse rotation instance.
        # See your notes for more information.
        orientation_inverse = self.orientation.inv()

        self.output = []
        for led in self.leds_visible:
            camera_to_led_inertial_coordinate = (
                led["inertial_coordinate"] - self.inertial_coordinates
            )

            camera_to_led_camera_coordinate = orientation_inverse.apply(
                camera_to_led_inertial_coordinate
            )

            x_image = (
                -self.focal_length * camera_to_led_camera_coordinate[0]
            ) / camera_to_led_camera_coordinate[2]
            y_image = (
                -(self.focal_length * camera_to_led_camera_coordinate[1])
                / camera_to_led_camera_coordinate[2]
            )
            # Negative sign because the camera is looking at -z in its body frame

            self.output.append((x_image, y_image, led["color"]))

    def get_image_vertex_coordinates(self):
        """
        This method computes the (x,y) coordinates of the 4 vertices of the image and stores them in the list self.image_vertex_coordinates
        The method used to compute is similar to the plot_camera_fov() function in PlotFunctions.py
        We get the length of FoV bounding lines which depend on the range of the camera, then find the end points of the FoV bounding lines.
        The end points are obtained in the 3D space in the camera body frame rather than the inertial frame in the plot_camera_fov() function.
        We then project those points into 2D space to get the (x,y) image plane coordinates of the vertices.
        This function needs to be called only when the range or FoV or focal_length is being changed, as the image size depends only on these camera intrinsic parameters.
        (I hope this method is correct!)"""

        self.image_vertex_coordinates = []

        for vector in self.body_frame_fov_bounding_vectors:
            normalized_vector = vector / np.linalg.norm(vector)
            fov_line_length = self.max_range / np.dot(
                self.body_frame_pointing_vector, normalized_vector
            )

            end_point = fov_line_length * normalized_vector
            x_coordinate = (self.focal_length * end_point[0]) / end_point[2]
            y_coordinate = (self.focal_length * end_point[1]) / end_point[2]
            self.image_vertex_coordinates.append((x_coordinate, y_coordinate))
