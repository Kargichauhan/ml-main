from scipy.spatial.transform import Rotation as R
import numpy as np


class cubesat:
    def __init__(self, size: str):
        """
        Generates a CubeSat with a Designation as User Specified, '1U' or '3U' or '6U', etc.
        Make the vertices of the CubeSat (in the body frame)
        Initializes at Identity Attitude with on vertex at the Inertial Frame Origin.
        The translate and rotate methods move the CubeSat and orient it according to user specifications.
        """

        self.size = size
        self.name = "default cubesat"

        self.orientation = R.identity()
        # We start from Identity Attitude #

        self.distance_from_inertial_origin = np.array([0.0, 0.0, 0.0])
        # We start with one vertex at the Inertial Frame Origin #

        self.body_frame_x_axis = np.array([1.0, 0.0, 0.0])
        self.body_frame_y_axis = np.array([0.0, 1.0, 0.0])
        self.body_frame_z_axis = np.array([0.0, 0.0, 1.0])
        self.inertial_frame_x_axis = self.body_frame_x_axis.copy()
        self.inertial_frame_y_axis = self.body_frame_y_axis.copy()  
        self.inertial_frame_z_axis = self.body_frame_z_axis.copy()

        self.body_frame_linear_velocity = np.array([0.0, 0.0, 0.0])
        self.inertial_frame_linear_velocity = self.body_frame_linear_velocity.copy()

        self.body_frame_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.inertial_frame_angular_velocity = self.body_frame_angular_velocity.copy()

        if self.size == "1U":
            self.body_frame_vertices = np.array(
                [
                    [0, 0, 0],
                    [0.10, 0, 0],
                    [0.10, 0.10, 0],
                    [0, 0.10, 0],
                    [0, 0, 0.10],
                    [0.10, 0, 0.10],
                    [0.10, 0.10, 0.10],
                    [0, 0.10, 0.10],
                ],
                dtype=float,
            )
            self.inertial_frame_vertices = self.body_frame_vertices.copy()

            self.body_frame_leds = np.array(
                [
                    [0.10, 0.0275, 0.0275],  # y
                    [0.10, 0.07, 0.0275],  # r
                    [0.10, 0.07, 0.07],  # g
                    [0.10, 0.0275, 0.07],  # b
                    [0.0275, 0, 0.0275],  # y
                    [0.07, 0, 0.0275],  # g
                    [0.07, 0, 0.07],  # b
                    [0.0275, 0, 0.07],  # r
                    [0.0275, 0.0275, 0],  # b
                    [0.0275, 0.07, 0],  # r
                    [0.07, 0.07, 0],  # g
                    [0.07, 0.0275, 0],  # y
                ],
                dtype=float,
            )
            self.inertial_frame_leds = self.body_frame_leds.copy()

        if self.size == "3U":
            self.body_frame_vertices = np.array(
                [
                    [0, 0, 0],
                    [0.10, 0, 0],
                    [0.10, 0.10, 0],
                    [0, 0.10, 0],
                    [0, 0, 0.30],
                    [0.10, 0, 0.30],
                    [0.10, 0.10, 0.30],
                    [0, 0.10, 0.30],
                ],
                dtype=float,
            )
            self.inertial_frame_vertices = self.body_frame_vertices.copy()

            self.body_frame_leds = np.array(
                [
                    [0.10, 0.025, 0.10],
                    [0.10, 0.075, 0.10],
                    [0.10, 0.075, 0.20],
                    [0.10, 0.025, 0.20],
                ],
                dtype=float,
            )
            self.inertial_frame_leds = self.body_frame_leds.copy()

        self.body_frame_normals_to_planes = np.array(
            [
                [0, 0, -1],
                [0, 0, 1],
                [-1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, -1, 0],
            ]
        )
        self.inertial_frame_normals_to_planes = self.body_frame_normals_to_planes.copy()

    def get_vertices(self):
        self.vertex_1 = {
            "name": "vertex 1",
            "body_coordinate": self.body_frame_vertices[0],
            "inertial_coordinate": self.inertial_frame_vertices[0],
        }
        self.vertex_2 = {
            "name": "vertex 2",
            "body_coordinate": self.body_frame_vertices[1],
            "inertial_coordinate": self.inertial_frame_vertices[1],
        }
        self.vertex_3 = {
            "name": "vertex 3",
            "body_coordinate": self.body_frame_vertices[2],
            "inertial_coordinate": self.inertial_frame_vertices[2],
        }
        self.vertex_4 = {
            "name": "vertex 4",
            "body_coordinate": self.body_frame_vertices[3],
            "inertial_coordinate": self.inertial_frame_vertices[3],
        }
        self.vertex_5 = {
            "name": "vertex 5",
            "body_coordinate": self.body_frame_vertices[4],
            "inertial_coordinate": self.inertial_frame_vertices[4],
        }
        self.vertex_6 = {
            "name": "vertex 6",
            "body_coordinate": self.body_frame_vertices[5],
            "inertial_coordinate": self.inertial_frame_vertices[5],
        }
        self.vertex_7 = {
            "name": "vertex 7",
            "body_coordinate": self.body_frame_vertices[6],
            "inertial_coordinate": self.inertial_frame_vertices[6],
        }
        self.vertex_8 = {
            "name": "vertex 8",
            "body_coordinate": self.body_frame_vertices[7],
            "inertial_coordinate": self.inertial_frame_vertices[7],
        }

        self.vertices_list = [
            self.vertex_1,
            self.vertex_2,
            self.vertex_3,
            self.vertex_4,
            self.vertex_5,
            self.vertex_6,
            self.vertex_7,
            self.vertex_8,
        ]

    def get_edges(self):
        """
        Generates all the 12 edges of the CubeSat in Body-Frame Vectors as attributes, and then generates a list of all the Edges.
        """
        self.get_vertices()

        self.edge_1to2 = {
            "start_point": self.vertex_1,
            "end_point": self.vertex_2,
            "inertial_coordinate": self.vertex_2["inertial_coordinate"]
            - self.vertex_1["inertial_coordinate"],
        }
        self.edge_2to3 = {
            "start_point": self.vertex_2,
            "end_point": self.vertex_3,
            "inertial_coordinate": self.vertex_3["inertial_coordinate"]
            - self.vertex_2["inertial_coordinate"],
        }
        self.edge_3to4 = {
            "start_point": self.vertex_3,
            "end_point": self.vertex_4,
            "inertial_coordinate": self.vertex_4["inertial_coordinate"]
            - self.vertex_3["inertial_coordinate"],
        }
        self.edge_4to1 = {
            "start_point": self.vertex_4,
            "end_point": self.vertex_1,
            "inertial_coordinate": self.vertex_1["inertial_coordinate"]
            - self.vertex_4["inertial_coordinate"],
        }
        self.edge_5to6 = {
            "start_point": self.vertex_5,
            "end_point": self.vertex_6,
            "inertial_coordinate": self.vertex_6["inertial_coordinate"]
            - self.vertex_5["inertial_coordinate"],
        }
        self.edge_6to7 = {
            "start_point": self.vertex_6,
            "end_point": self.vertex_7,
            "inertial_coordinate": self.vertex_7["inertial_coordinate"]
            - self.vertex_6["inertial_coordinate"],
        }
        self.edge_7to8 = {
            "start_point": self.vertex_7,
            "end_point": self.vertex_8,
            "inertial_coordinate": self.vertex_8["inertial_coordinate"]
            - self.vertex_7["inertial_coordinate"],
        }
        self.edge_8to5 = {
            "start_point": self.vertex_8,
            "end_point": self.vertex_5,
            "inertial_coordinate": self.vertex_5["inertial_coordinate"]
            - self.vertex_8["inertial_coordinate"],
        }
        self.edge_1to5 = {
            "start_point": self.vertex_1,
            "end_point": self.vertex_5,
            "inertial_coordinate": self.vertex_5["inertial_coordinate"]
            - self.vertex_1["inertial_coordinate"],
        }
        self.edge_2to6 = {
            "start_point": self.vertex_2,
            "end_point": self.vertex_6,
            "inertial_coordinate": self.vertex_6["inertial_coordinate"]
            - self.vertex_2["inertial_coordinate"],
        }
        self.edge_3to7 = {
            "start_point": self.vertex_3,
            "end_point": self.vertex_7,
            "inertial_coordinate": self.vertex_7["inertial_coordinate"]
            - self.vertex_3["inertial_coordinate"],
        }
        self.edge_4to8 = {
            "start_point": self.vertex_4,
            "end_point": self.vertex_8,
            "inertial_coordinate": self.vertex_8["inertial_coordinate"]
            - self.vertex_4["inertial_coordinate"],
        }

        self.edges_list = [
            self.edge_1to2,
            self.edge_2to3,
            self.edge_3to4,
            self.edge_4to1,
            self.edge_5to6,
            self.edge_6to7,
            self.edge_7to8,
            self.edge_8to5,
            self.edge_1to5,
            self.edge_2to6,
            self.edge_3to7,
            self.edge_4to8,
        ]

    def get_leds(self):
        """
        Generates all the LEDs as Dictionaries as attributes, then generates a list of all the LEDs
        """
        self.get_planes()

        self.led_1 = {
            "name": "LED 1",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[0],
            "inertial_coordinate": self.inertial_frame_leds[0],
            "plane": self.plane_5,
            "color": "yellow",
            "cameras_that_can_see": [],
        }
        self.led_2 = {
            "name": "LED 2",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[1],
            "inertial_coordinate": self.inertial_frame_leds[1],
            "plane": self.plane_5,
            "color": "red",
            "cameras_that_can_see": [],
        }
        self.led_3 = {
            "name": "LED 3",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[2],
            "inertial_coordinate": self.inertial_frame_leds[2],
            "plane": self.plane_5,
            "color": "green",
            "cameras_that_can_see": [],
        }
        self.led_4 = {
            "name": "LED 4",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[3],
            "inertial_coordinate": self.inertial_frame_leds[3],
            "plane": self.plane_5,
            "color": "blue",
            "cameras_that_can_see": [],
        }
        self.led_5 = {
            "name": "LED 5",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[4],
            "inertial_coordinate": self.inertial_frame_leds[4],
            "plane": self.plane_6,
            "color": "yellow",
            "cameras_that_can_see": [],
        }
        self.led_6 = {
            "name": "LED 6",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[5],
            "inertial_coordinate": self.inertial_frame_leds[5],
            "plane": self.plane_6,
            "color": "green",
            "cameras_that_can_see": [],
        }
        self.led_7 = {
            "name": "LED 7",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[6],
            "inertial_coordinate": self.inertial_frame_leds[6],
            "plane": self.plane_6,
            "color": "blue",
            "cameras_that_can_see": [],
        }
        self.led_8 = {
            "name": "LED 8",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[7],
            "inertial_coordinate": self.inertial_frame_leds[7],
            "plane": self.plane_6,
            "color": "red",
            "cameras_that_can_see": [],
        }
        self.led_9 = {
            "name": "LED 9",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[8],
            "inertial_coordinate": self.inertial_frame_leds[8],
            "plane": self.plane_1,
            "color": "blue",
            "cameras_that_can_see": [],
        }
        self.led_10 = {
            "name": "LED 10",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[9],
            "inertial_coordinate": self.inertial_frame_leds[9],
            "plane": self.plane_1,
            "color": "red",
            "cameras_that_can_see": [],
        }
        self.led_11 = {
            "name": "LED 11",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[10],
            "inertial_coordinate": self.inertial_frame_leds[10],
            "plane": self.plane_1,
            "color": "green",
            "cameras_that_can_see": [],
        }
        self.led_12 = {
            "name": "LED 12",
            "parent_cubesat": self,
            "body_coordinate": self.body_frame_leds[11],
            "inertial_coordinate": self.inertial_frame_leds[11],
            "plane": self.plane_1,
            "color": "yellow",
            "cameras_that_can_see": [],
        }

        self.leds_list = [
            self.led_1,
            self.led_2,
            self.led_3,
            self.led_4,
            self.led_5,
            self.led_6,
            self.led_7,
            self.led_8,
            self.led_9,
            self.led_10,
            self.led_11,
            self.led_12,
        ]

        # # Use the below list if you want only one face of LEDs
        # self.leds_list = [
        #     self.led_1,
        #     self.led_2,
        #     self.led_3,
        #     self.led_4,
        # ]

    def get_planes(self):
        """
        Generates all the planes as Dictionaries of a normal planes and points as attributes, then generates a list of all the planes
        """
        self.plane_1 = {
            "name": "plane 1",
            "parent_cubesat": self,
            "plane_body_coordinate": self.body_frame_normals_to_planes[0],
            "plane_inertial_coordinate": self.inertial_frame_normals_to_planes[0],
            "points": [self.vertex_1, self.vertex_2, self.vertex_3, self.vertex_4],
        }
        self.plane_2 = {
            "name": "plane 2",
            "parent_cubesat": self,
            "plane_body_coordinate": self.body_frame_normals_to_planes[1],
            "plane_inertial_coordinate": self.inertial_frame_normals_to_planes[1],
            "points": [self.vertex_5, self.vertex_6, self.vertex_7, self.vertex_8],
        }
        self.plane_3 = {
            "name": "plane 3",
            "parent_cubesat": self,
            "plane_body_coordinate": self.body_frame_normals_to_planes[2],
            "plane_inertial_coordinate": self.inertial_frame_normals_to_planes[2],
            "points": [self.vertex_1, self.vertex_4, self.vertex_8, self.vertex_5],
        }
        self.plane_4 = {
            "name": "plane 4",
            "parent_cubesat": self,
            "plane_body_coordinate": self.body_frame_normals_to_planes[3],
            "plane_inertial_coordinate": self.inertial_frame_normals_to_planes[3],
            "points": [self.vertex_3, self.vertex_7, self.vertex_8, self.vertex_4],
        }
        self.plane_5 = {
            "name": "plane 5",
            "parent_cubesat": self,
            "plane_body_coordinate": self.body_frame_normals_to_planes[4],
            "plane_inertial_coordinate": self.inertial_frame_normals_to_planes[4],
            "points": [self.vertex_3, self.vertex_7, self.vertex_6, self.vertex_2],
        }
        self.plane_6 = {
            "name": "plane 6",
            "parent_cubesat": self,
            "plane_body_coordinate": self.body_frame_normals_to_planes[5],
            "plane_inertial_coordinate": self.inertial_frame_normals_to_planes[5],
            "points": [self.vertex_2, self.vertex_6, self.vertex_5, self.vertex_1],
        }
        self.planes_list = [
            self.plane_1,
            self.plane_2,
            self.plane_3,
            self.plane_4,
            self.plane_5,
            self.plane_6,
        ]

    def rotate(self, rotation_instance, rotation_frame="inertial"):
        """
        This method rotates the CubeSat by "rotation_instance".
        It also stores the cumulative rotations (Current Attitude)
        The "rotation_instance" input argument should be a "Rotation" instance from scipy.spatial.transform
        Make sure to note that the rotation is a Inertial-Frame rotation.
        The "rotation_instance" uses q-convention [vector*sin(), angle*cos()], beware.
        """

        # So the methodology being followed is this:
        # We record the attitude of the body by composing the previous attitude with the rotation_instance to be applied now.
        # Then, we apply the rotation by the new attitude.
        # Then, we add the new attitude to the distance from the origin to place the body at the correct distance in the inertial frame
        # We have to perform the rotation about the inertial origin only, as the rotation method works that way
        # Uses q-convention [vector*sin(), angle*cos()], Beware!!!

        if rotation_frame == "body":
            rotvec_inertial_frame = rotation_instance.as_rotvec()
            rotvec_body_frame = self.orientation.apply(rotvec_inertial_frame)
            # We now have converted the rotation vector from the body frame to the inertial frame.
            # The scipy rotation library only allows rotations about the inertial frame.
            rotation_instance_body_frame = R.from_rotvec(rotvec_body_frame)  # type: ignore
            self.orientation = rotation_instance_body_frame * self.orientation

            # Perform rotation about inertial origin and move body by distance from inertial origin #
            self.inertial_frame_vertices = (
                self.distance_from_inertial_origin
                + self.orientation.apply(self.body_frame_vertices)
            )
            self.inertial_frame_leds = (
                self.distance_from_inertial_origin
                + self.orientation.apply(self.body_frame_leds)
            )
            self.inertial_frame_normals_to_planes = self.orientation.apply(
                self.body_frame_normals_to_planes
            )
            self.inertial_frame_x_axis = self.orientation.apply(self.body_frame_x_axis)
            self.inertial_frame_y_axis = self.orientation.apply(self.body_frame_y_axis)
            self.inertial_frame_z_axis = self.orientation.apply(self.body_frame_z_axis)

            self.inertial_frame_linear_velocity = self.orientation.apply(
                self.body_frame_linear_velocity
            )
            self.inertial_frame_angular_velocity = self.orientation.apply(
                self.body_frame_angular_velocity
            )

            self.get_vertices()
            self.get_edges()
            self.get_planes()
            self.get_leds()

        else:
            # This rotation composition takes the previous rotation state, and composes another rotation to it to keep track #
            self.orientation = rotation_instance * self.orientation
            # self.orientation is the current rotation instance of the camera body wrt the inertial frame.

            # Perform rotation about inertial origin and move body by distance from inertial origin #
            self.inertial_frame_vertices = (
                self.distance_from_inertial_origin
                + self.orientation.apply(self.body_frame_vertices)
            )
            self.inertial_frame_leds = (
                self.distance_from_inertial_origin
                + self.orientation.apply(self.body_frame_leds)
            )
            self.inertial_frame_normals_to_planes = self.orientation.apply(
                self.body_frame_normals_to_planes
            )
            self.inertial_frame_x_axis = self.orientation.apply(self.body_frame_x_axis)
            self.inertial_frame_y_axis = self.orientation.apply(self.body_frame_y_axis)
            self.inertial_frame_z_axis = self.orientation.apply(self.body_frame_z_axis)

            self.inertial_frame_linear_velocity = self.orientation.apply(
                self.body_frame_linear_velocity
            )
            self.inertial_frame_angular_velocity = self.orientation.apply(
                self.body_frame_angular_velocity
            )

            self.get_vertices()
            self.get_edges()
            self.get_planes()
            self.get_leds()

    def translate(self, distance):
        """
        This method moves the CubeSat by "distance".
        It also stores the cumulative distance moved by the CubeSat in all three coordinates.
        The "distance" input argument should be a 3-array like.
        """

        distance = np.array(distance).reshape((3,))

        self.distance_from_inertial_origin = (
            self.distance_from_inertial_origin + distance
        )  # Adding new translation to the previous distance to keep track #

        self.inertial_frame_vertices = self.inertial_frame_vertices + distance
        self.inertial_frame_leds = self.inertial_frame_leds + distance

        self.get_vertices()
        self.get_edges()
        self.get_planes()
        self.get_leds()

    def move_to(self, inertial_coordinates):
        """
        This method just moves the cubesat object to a particular coordinate instead of translating it.
        """
        distance = np.array(inertial_coordinates) - self.inertial_frame_vertices[0]
        self.translate(distance)

    def set_inertial_frame_velocity(
        self, desired_inertial_linear_velocity, desired_inertial_angular_velocity
    ):
        """
        This is how you set the velocity of the body in the inertial frame.
        If you simply try to change the inertial_frame_velocity parameter, it won't work.
        The inertial_frame_velocities are parameters that get recomputed based on the body_frame_velocities constantly.
        This function takes the desired inertial_frame_velocities, converts them to body frame velocities and then sets the body_frame_velocities accordingly.
        IMPORTANT:
        If both translational and rotational velocities are being used in a simulation, then the propagator should set the velocities at each and every time step, especially the linear velocity.
        "Rotational velocity" about the inertial frame does not mean "Revolution" of the body about the Inertial Frame. It merely means the rotation of the body, but about the inertial frame axes.
        """
        orientation_inverse = self.orientation.inv()
        # self.orientation takes body frame and converts it into inertial frame.
        # So orientation_inverse should take inertial frame vectors and convert to body frame.

        self.body_frame_linear_velocity = orientation_inverse.apply(
            np.array(desired_inertial_linear_velocity)  # type: ignore
        )
        self.body_frame_angular_velocity = orientation_inverse.apply(
            np.array(desired_inertial_angular_velocity)
        )

        self.rotate(R.identity())
