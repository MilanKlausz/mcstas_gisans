
import numpy as np

storedDataParameters = {
  'D22_100nm_Air_GISANS' : {
    'filePath' : '/Users/milanklausz/Desktop/MultiDetector1_data.csv',
    'xLimits' : [-0.304, 0.304],
    'yLimits' : [-0.128, 0.481],
    'experimentTime' : 10800 #sec
  }
}

# def readBornAgainIntFile(filePath):
#   data = []
#   with open(filePath, 'r') as file:
#     for line in file:
#       if '# data' in line:
#         for data_line in file:
#           data_line = data_line.strip()
#           if data_line:
#             data_values = data_line.split()
#             data.append([float(value) for value in data_values])
#           else:
#             break
#   return np.array(data)

def readCsvFile(filePath):
  data = []
  with open(filePath, 'r') as file:
    for data_line in file:
      data_line = data_line.strip()
      if data_line:
        data_values = data_line.split(",")
        data.append([float(value) for value in data_values])
      else:
        break
  return np.array(data)

def getStoredData(dataId):
  filePath = storedDataParameters[dataId]['filePath']
  xLimits = storedDataParameters[dataId]['xLimits']
  yLimits = storedDataParameters[dataId]['yLimits']
  experimentTime = storedDataParameters[dataId]['experimentTime']

  data = readCsvFile(filePath)
  print(data.shape)
  error = np.sqrt(data) #Assuming sqrt(N) error
  xEdges = np.linspace(xLimits[0], xLimits[1], data.shape[1]+1)
  zEdges = np.linspace(yLimits[0], yLimits[1], data.shape[0]+1)
  # xEdges = np.linspace(xLimits[0], xLimits[1], data.shape[0]+1)
  # zEdges = np.linspace(yLimits[0], yLimits[1], data.shape[1]+1)

  return data, error, xEdges, zEdges, experimentTime
