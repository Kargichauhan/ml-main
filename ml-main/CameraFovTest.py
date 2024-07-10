from CamFunctions import *

# This file is used to test the algorithm for the camera FoV test.
# The algorithm is present in the slides. It is linear programming. The code is present in CamFunctions.py
# Methodology:
# We have an excel sheet with the following: Camera coordinates, Camera FoV angles, FoV vectors, particle coordinates, expected result.
# Open and read the excel sheet, then take the Camera coordinates, FoV angles, particle coordinates.
# We spawn a camera with the given FoV, spawn a particle with the coordinates that is read from the excel sheet.
# Then, apply the algorithm to find out if the particle is within the FoV of the camera.
# Output the results to a CSV file: Camera coordinates, Camera FoV angles, FoV vectors, particle coordinates, algorithm result.
# Finally, compare the true result and the algorithm result and record it in the excel sheet.
