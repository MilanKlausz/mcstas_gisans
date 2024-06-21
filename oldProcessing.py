
import numpy as np
from neutron_utilities import V2L, tofToLambda
from importlib import import_module

def virtualPropagationToDetector(x, y, z, vx, vy, vz, rot_matrix, sample_detector_distance):
  """Calculate x,y,z position on the detector surface and the corresponding tof for the sample to detector propagation"""
  #compensate coord system rotation
  _, yRot = np.dot(rot_matrix, [z, y])
  _, vyRot = np.dot(rot_matrix, [vz, vy])

  #propagate to detector surface perpendicular to the y-axis
  t_propagate= (sample_detector_distance - yRot) / vyRot

  x = x + vx * t_propagate
  y = y + vy * t_propagate
  z = z + vz * t_propagate

  return x, y, z, t_propagate

def qConvFactorFromTofAtDetector(sample_detector_path_length, nominal_source_sample_distance, tDet):
  path_length = nominal_source_sample_distance + sample_detector_path_length
  return 2*np.pi/(tofToLambda(tDet, path_length)*0.1)

def processNeutronsNonVectorised(events, get_simulation, sc):
  sim_module=import_module(sc['sim_module_name'])
  get_sample=sim_module.get_sample

  misses = 0
  average_alpha_i = 0 #sanity check
  total = len(events)
  out_events = []
  q_events_real = [] #p,qx,qy,qz,t - calculated with lambda and true incident and outgoing directions
  q_events_no_incident_info = [] #p,qx,qy,qz,t - calculated with lambda but not using incident direction
  q_events_calc_lambda = [] #p,qx,qy,qz,t - calculated from TOF but using true incident direction
  q_events_calc_sample = [] #p,qx,qy,qz,t - calculated from TOF at sample position
  q_events_calc_detector = [] #p,qx,qy,qz,t - calculated from TOF at the detector position

  rot_matrix = np.array([[np.cos(sc['alpha_inc']), -np.sin(sc['alpha_inc'])],[np.sin(sc['alpha_inc']), np.cos(sc['alpha_inc'])]])
  v_in_alpha = np.array([0, np.cos(sc['alpha_inc']), np.sin(sc['alpha_inc'])])

  notTOFInstrument = sc['wavelengthSelected'] is not None # just to make the code more readable later on
  qConvFactorFixed = None if notTOFInstrument is False else 2*np.pi/(sc['wavelengthSelected']*0.1)

  for in_ID, neutron in enumerate(events):
    if in_ID%200==0:
      print(f'{in_ID:10}/{total}')
    p, x, y, z, vx, vy, vz, t = neutron
    alpha_i = np.arctan(vz/vy)*180./np.pi  # deg
    average_alpha_i += alpha_i
    phi_i = np.arctan(vx/vy)*180./np.pi  # deg
    v = np.sqrt(vx**2+vy**2+vz**2)
    wavelength = V2L/v  # Å
    qConvFactorFromLambda = 2*np.pi/(wavelength * 0.1)
    qConvFactorFromTof = qConvFactorFixed if notTOFInstrument else 2*np.pi/(tofToLambda(t, sc['nominal_source_sample_distance'])*0.1) #for an intermediate result
    v_in = np.array([vx, vy, vz]) / v

    sample = get_sample(radius=sc['silicaRadius'])

    #calculate pixelNr² outgoing beams with a random angle within one pixel range
    Ry = 2*np.random.random()-1
    Rz = 2*np.random.random()-1
    sim = get_simulation(sample, sc['pixelNr'], sc['angle_range'], wavelength, alpha_i, p, Ry, Rz)
    sim.options().setUseAvgMaterials(True)
    sim.options().setIncludeSpecular(True)
    res = sim.simulate()
    # get probability (intensity) for all pixels
    pout = res.array()
    # calculate beam angle relative to coordinate system, including incident beam direction
    alpha_f = sc['angle_range']*(np.linspace(1., -1., sc['pixelNr'])+Ry/(sc['pixelNr']-1))
    phi_f = phi_i+sc['angle_range']*(np.linspace(-1., 1., sc['pixelNr'])+Rz/(sc['pixelNr']-1))
    alpha_f_rad = alpha_f * np.pi/180.
    phi_f_rad = phi_f * np.pi/180.
    alpha_grid, phi_grid = np.meshgrid(alpha_f_rad, phi_f_rad)

    VX_grid = v * np.cos(alpha_grid) * np.sin(phi_grid)
    VY_grid = v * np.cos(alpha_grid) * np.cos(phi_grid)
    VZ_grid = -v * np.sin(alpha_grid)

    for pouti, vxi, vyi, vzi in zip(pout.T.flatten(), VX_grid.flatten(),  VY_grid.flatten(), VZ_grid.flatten()):
      out_events.append([pouti, x, y, z, vxi, vyi, vzi, t])
      v_out = np.array([vxi, vyi, vzi]) / v
      q_events_real.append([pouti, *(qConvFactorFromLambda * np.subtract(v_out, v_in)), t])

      q_events_no_incident_info.append([pouti, *(qConvFactorFromLambda * np.subtract(v_out, v_in_alpha)), t])
      q_events_calc_lambda.append([pouti, *(qConvFactorFromTof * np.subtract(v_out, v_in)), t])

      q_events_calc_sample.append([pouti, *(qConvFactorFromTof * np.subtract(v_out, v_in_alpha)), t])

      xDet, yDet, zDet, sample_detector_tof = virtualPropagationToDetector(x, y, z, vxi, vyi, vzi, rot_matrix, sc['sample_detector_distance'])
      sample_detector_path_length = np.linalg.norm([xDet, yDet, zDet])
      v_out_det = np.array([xDet, yDet, zDet]) / sample_detector_path_length
      qConvFactorFromTofAtDet = qConvFactorFixed if notTOFInstrument else qConvFactorFromTofAtDetector(sample_detector_path_length, sc['nominal_source_sample_distance'], t+sample_detector_tof)
      q_events_calc_detector.append([pouti, *(qConvFactorFromTofAtDet * np.subtract(v_out_det, v_in_alpha)), t])

  print("misses:", misses)
  print(f"average_alpha_i: {average_alpha_i/len(events)}")
  return np.array(out_events), np.array(q_events_real), np.array(q_events_no_incident_info), np.array(q_events_calc_lambda), np.array(q_events_calc_sample), np.array(q_events_calc_detector)
