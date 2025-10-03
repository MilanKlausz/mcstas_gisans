"""
Contains detector related calculations
"""

import numpy as np

from .neutron_calculations import calculate_wavelength, qConvFactor

def detector_plane_intersection(x, y, z, VX, VY, VZ, sample_detector_distance, sampleToRealCoordRotMatrix, realToSampleCoordRotMatrix):
  """
  Calculate x,y,z position on the detector surface and the corresponding TOF
  for the sample to detector propagation
  """
  #Calculate time until the detector surface with coord system rotation
  # NOTE: under the assumption that the detector surface is vertical in the real coord system
  zRot, _ = np.dot(sampleToRealCoordRotMatrix, [z, y])
  vzRot, _ = np.matmul(sampleToRealCoordRotMatrix, np.vstack((VZ, VY))) # get [vz, vy]

  #propagate to detector surface perpendicular to the y-axis
  t_propagate = (sample_detector_distance - zRot) / vzRot

  #Calculate the effect of gravity
  gravityAcceleration = 9.80665 #m/s2
  zGravityAcc, yGravityAcc = np.dot(realToSampleCoordRotMatrix, [0, -gravityAcceleration]) #TODO should not happen here
  zGravityDrop = zGravityAcc * 0.5 * t_propagate**2
  yGravityDrop = yGravityAcc * 0.5 * t_propagate**2

  return t_propagate, (VX * t_propagate + x), (VY * t_propagate + y + yGravityDrop), (VZ * t_propagate + z + zGravityDrop)

def calculate_detection_coordinate(xDet, yDet, zDet, sampleToRealCoordRotMatrix, realToSampleCoordRotMatrix):
  """
  Get the coordinate of the detection event from the position where the path of
  the particle intersects the plane of the detector surface. Using the exact
  position of intersection means infinite detector resolution.
  """
  #TODO generalise:
  #  pixel_size_x, pixel_size_y, sigma_x, sigma_y should be input

  detector_size_x = 1.024 #[m]
  detector_size_y = 1.024 #[m]
  detector_pixel_number_x = 256
  detector_pixel_number_y = 128
  pixel_size_x = detector_size_x / detector_pixel_number_x
  pixel_size_y = detector_size_y / detector_pixel_number_y

  # transform from the sample-based to the real coordinate system
  zDetReal, yDetReal = np.matmul(sampleToRealCoordRotMatrix, np.vstack((zDet, yDet)))
  #note: zDetReal is a fixed value due to the propagation to detector surface

  # apply gaussian randomisation
  sigma_x = 0.004/2.355 #FWHM=2.5-5[mm]
  sigma_y = 0 #0.005/2.355 #FWHM=2.5-5[mm]
  xDet = np.random.normal(xDet, sigma_x, size=xDet.shape)
  yDetReal = np.random.normal(yDetReal, sigma_y, size=yDetReal.shape)

  #get the coordinates of the centre of the pixel where the particle is detected
  xDetCoord = np.floor(xDet / pixel_size_x) * pixel_size_x + 0.5*pixel_size_x
  yDetCoordReal = np.floor(yDetReal / pixel_size_y) * pixel_size_y + 0.5*pixel_size_y

  #transform from the real to the sample-based coordinate system
  zDetCoord, yDetCoord = np.matmul(realToSampleCoordRotMatrix, np.vstack((zDetReal, yDetCoordReal)))

  return xDetCoord, yDetCoord, zDetCoord

def calculate_q(x, y, z, t, VX, VY, VZ, incident_direction, params):
  """
  Calculate Q values (x,y,z) from positions at the detector surface.
  All outgoing directions from the BornAgain simulation of a single particle are
  handled at the same time using operations on vectors.
  - Outgoing direction is calculated by propagating particles to the detector surface,
  and assuming that the particle is scattered at the centre of the sample (the origin).
  - Incident direction is an input value.
  - For non-TOF instruments the (2*pi/(wavelength)) factor (qConvFactorFixed) is an input value.
    For TOF instruments this factor is calculated from the TOF at the detector surface position
    and the nominal distance travelled by the particle until that position.
  """

  sampleToRealCoordRotMatrix = params['sampleToRealCoordRotMatrix']
  realToSampleCoordRotMatrix = params['realToSampleCoordRotMatrix']

  sample_detector_tof, x_detector_plane, y_detector_plane, z_detector_plane = detector_plane_intersection(x, y, z, VX, VY, VZ, params['sample_detector_distance'], sampleToRealCoordRotMatrix, realToSampleCoordRotMatrix)
  x_detection, y_detection, z_detection = calculate_detection_coordinate(x_detector_plane, y_detector_plane, z_detector_plane, sampleToRealCoordRotMatrix, realToSampleCoordRotMatrix)
  detection_coordinate = np.vstack((x_detection, y_detection, z_detection)).T
  sample_detector_path_length = np.linalg.norm(detection_coordinate, axis=1)

  outgoing_direction = detection_coordinate / sample_detector_path_length[:, np.newaxis]

  if params['wavelength_selected'] is None: #TOF instruments
    wavelength = calculate_wavelength(t + sample_detector_tof, params['nominal_source_sample_distance'] + sample_detector_path_length)
    qFactor = qConvFactor(wavelength)[:, np.newaxis]
  else: #not TOF instruments
    qFactor = qConvFactor(params['wavelength_selected'])

  return (outgoing_direction - incident_direction) * qFactor