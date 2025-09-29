#-*- coding: utf-8 -*-
'''
Simple support library for reading, analyzing and plotting of McStas results
from the Estia instrument simulations.
Meant to be used from IPython Notebook or QtConsole, but can also be run stand alone.
'''

import os, sys
from numpy import *

try:
    import h5py
except ImportError:
    print("h5py not found, modern NeXuS format will not be readable.")

try:
  from IPython import display #@UnusedImport
  from IPython.core.pylabtools import print_figure
  from matplotlib.figure import Figure
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.colors import LogNorm
except ImportError:
  # IPython and/or matplotlib not available, no interactive functionalities
  display=None

MAX_EVTS_BATCH=50000
    
class McSim(object):
  '''
  Object representing complete simulation from McStas.
  Supports either the old style McStas output with separate ASCII files 
  or HDF5 single file output.
  
  Different monitors can be accessed as keys like in a dictionary. The data is only loaded when the monitor is first accessed.
  '''

  def __init__(self, path):
    '''
    Initialize the simulation. path should be the file name to either the NeXuS or the mccode.sim file.
    '''
    self._data={}
    if path.endswith('.h5'):
      self._init_hdf(path)
    elif path.endswith('.sim'):
      self._init_old(path)
    else:
      if os.path.exists(os.path.join(path,'mccode.h5')):
        self._init_hdf(os.path.join(path,'mccode.h5'))
      elif os.path.exists(os.path.join(path, 'mccode.sim')):
        self._init_old(os.path.join(path, 'mccode.sim'))
      else:
        raise IOError("Can't locate mccode.h5 of mccode.sim file in %s"%path)

  def _init_hdf(self, path):
    self.hdf=h5py.File(path, 'r')
    self.data_loader=DataLoaderHDF(self.hdf)
    self.info=self.data_loader.info

  def _init_old(self, path):
    self.info=HeaderFile(path)
    self.data_loader=DataLoaderOld(self.info, os.path.dirname(path))

  def keys(self):
    return list(self.info['data'].keys())

  def monitors(self):
    output=[]
    for val in list(self.info['data'].values()):
      xy=(val['xvar'], val['yvar'])
      if not xy in output:
        output.append(xy)
    output.sort()
    return output
  
  def plot(self, monitors=None):
    graphs={}
    for key, val in list(self.info['data'].items()):
      xy=(val['xvar'], val['yvar'])
      if monitors is not None and xy!=monitors:
        continue
      data=self[key]
      if xy in graphs:
        graphs[xy].append(data)
      else:
        graphs[xy]=[data]
    for xy, datasets in sorted(graphs.items()):
      cols=min(len(datasets), 3)
      rows=len(datasets)/3+1

      fig=Figure(figsize=(12, 5*rows), dpi=300, facecolor='#FFFFFF')
      FigureCanvasAgg(fig)

      for i, data in enumerate(datasets):
        ax=fig.add_subplot(rows, cols, i+1)
        data.plot(ax=ax)
      graphs[xy]=fig
    if monitors is None:
      return graphs
    else:
      return fig

  def __getitem__(self, item):
    if item in self._data:
      return self._data[item]
    elif item in list(self.keys()):
      data=self.data_loader.load_item(item)
      self._data[item]=data
      return data
    else:
      raise KeyError("Can't find dataset %s"%item)

class HeaderFile(object):
  '''
  Analyze McStas mccode.sim header files.
  '''
  _data=None

  @property
  def data(self):
    return self._data['data']

  def __init__(self, path):
    self._data={}
    data=open(path, 'r').read()
    data_lines=data.splitlines()
    if not data.startswith('McStas simulation description file'):
      raise IOError('Not a valid McStas description file.')
    self._data['start_time']=data_lines[1].split(':', 1)[1].strip()
    self._data['program_name']=data_lines[2].split(':', 1)[1].strip()
    self._data['data']=self.get_data(data)

  def get_data(self, data):
    output={}
    start_idx=0
    while start_idx<len(data):
      start_idx=data.find('begin data', start_idx)
      if start_idx==-1:
        break
      end_idx=data.find('end data', start_idx)
      block=data[start_idx:end_idx].splitlines()[1:]
      block_info={}
      for bline in block:
        item, value=bline.strip().split(':', 1)
        block_info[item]=value.strip()
        if item=='component':
          output[value.strip()]=block_info

      start_idx=end_idx
    return output

  def __getitem__(self, item):
    return self._data[item]

class DataLoaderOld(object):
  '''
    Load and analyze a old style McStas format with a simulation and a set of data text files.
  '''
  def __init__(self, info, root):
    self.info=info
    self.root=root

  def load_item(self, item):
    item_info=self.info['data'][item]
    if not item_info['type'].startswith('array_2d'):
      return self.load_item_1d(item)

    fname=os.path.join(self.root, item_info['filename'])
    x_col=item_info['xvar']
    y_col=item_info['yvar']
    if x_col.startswith('Li') and y_col=='p': # Detector_nD     
      cols=item_info['variables'].split() 
      data=loadtxt(fname, dtype={'names': cols, 'formats': ['f4']*len(cols)})
      return TofData(data, item_info)
    else:
      raw=loadtxt(fname)
      data=raw[:len(raw)//3]
      return Dataset(data, item_info)

  def load_item_1d(self, item):
    item_info=self.info['data'][item]
    fname=os.path.join(self.root, item_info['filename'])
    raw=loadtxt(fname).T
    data=raw[1]
    errors=raw[2]
    return Dataset1D(data, errors, item_info)

class DataLoaderHDF(object):
  '''
    Load and analyze a new style NeXuS file format.
  '''
  def __init__(self, hdf):
    self.hdf=hdf['entry1']
    self.info={}
    self.info['data']={}
    for item in list(self.hdf['data'].keys()):
      node=self.hdf['data/'+item]
      info={}
      for key, value in list(node.attrs.items()):
        info[key.strip()]=value.decode('utf-8').strip()
      if info['filename'][-4:]=='.dat' or '_list.' in info['filename']:
        self.info['data'][info['component']]=info
      else:
        self.info['data'][info['filename']]=info
      info['datapath']='data/'+item

  def load_item(self, item):
    item_info=self.info['data'][item]
    if not item_info['type'].startswith('array_2d'):
      return self.load_item_1d(item)

    node=self.hdf[item_info['datapath']]
    x_col=item_info['xvar']
    y_col=item_info['yvar']
    if x_col.startswith('Li') and y_col=='p': # Detector_nD
      cols=item_info['variables'].split()
      evds=node['events']
      if len(evds)<=MAX_EVTS_BATCH:
        data=evds.value.astype(float32).view(
            dtype={'names': cols, 'formats': ['f4']*len(cols)}).flatten()
      else:
        ds=[]
        sys.stdout.write('Reading large dataset:\n')
        for i in range(len(evds)//MAX_EVTS_BATCH+1):
          sys.stdout.write('\r%i/%i'%(i*MAX_EVTS_BATCH, len(evds)))
          sys.stdout.flush()
          ds.append(evds[i*MAX_EVTS_BATCH:(i+1)*MAX_EVTS_BATCH])
        sys.stdout.write('\r%i/%i\n'%(len(evds), len(evds)))
        data=vstack(ds).astype(float32).view(
            dtype={'names': cols, 'formats': ['f4']*len(cols)}).flatten()
      return TofData(data, item_info)
    else:
      data=node['data'].value.T
      return Dataset(data, item_info)
  
  def load_item_1d(self, item):
    item_info=self.info['data'][item]
    node=self.hdf[item_info['datapath']]
    data=node['data'].value
    errors=node['errors'].value
    return Dataset1D(data, errors, item_info)

class Dataset1D(object):
  '''
    Representation of a standard McStas dataset of one variable.
  '''
  def __init__(self, data, errors, info):
    self.data=data
    self.errors=errors
    self.info=info

  def plot(self, log=False, ax=None):
    if ax is None:
      import pylab
      ax=pylab.gca()

    limits=list(map(float, self.info['xlimits'].split()))
    x=linspace(limits[0], limits[1], len(self.data))

    ax.errorbar(x, self.data, yerr=self.errors)
    if log:
      ax.set_yscale('log')
    else:
      ax.set_yscale('linear')
    ax.set_xlabel(self.info['xlabel'])
    ax.set_ylabel(self.info['ylabel'])
    ax.set_title(self.info['component'])

  def _repr_png_(self):
    '''
      Image representation form ipython console/notebook. 
    '''
    fig=Figure(figsize=(8, 5), dpi=300, facecolor='#FFFFFF')
    FigureCanvasAgg(fig)
    ax=fig.add_subplot(111)

    self.plot(ax=ax)

    return print_figure(fig, dpi=72)


class Dataset(object):
  '''
    Representation of a standard McStas dataset with 2D view.
  '''
  def __init__(self, data, info):
    self.data=data
    self.info=info

  def plot(self, ax=None, cbar=False, **kwargs):
    if ax is None:
      import pylab
      ax=pylab.gca()

    limits=list(map(float, self.info['xylimits'].split()))

    img=ax.imshow(self.data, origin='lower', extent=limits, aspect='auto', **kwargs)

    ax.set_xlabel(self.info['xlabel'])
    ax.set_ylabel(self.info['ylabel'])
    ax.set_title(self.info['component'])

    if cbar:
      pylab.colorbar(img, label='Intensity')

  def _repr_png_(self):
    '''
      Image representation form ipython console/notebook. 
    '''
    fig=Figure(figsize=(8, 5), dpi=300, facecolor='#FFFFFF')
    FigureCanvasAgg(fig)
    ax=fig.add_subplot(111)

    self.plot(ax=ax)

    return print_figure(fig, dpi=72)

class TofData(Dataset):
  '''
    Representation of a dataset collected with Monitor_nD
  '''
  
  def project1d(self, col, bins=50, fltr=None, newcols=None, norm=None, errors=False):
    ''' 
      Generate binned data for arbitrary binning and columns.
      
      The fltr argument can be a string with a filtering condition on the available columns.
      Syntax for this filtering follows numpy array convetions like (x>5)&(abs(L)<0.3) would
      be a valid statement.
      
      newcols can be a list of (name, code) tuples, that calculate new columns from
      existing ones, like newcols=[('q', '4*pi/L*sin(theta)')].
    '''
    columns=dict([(coli, self.data[coli]) for coli in self.data.dtype.names])
    if newcols is not None:
      for name, code in newcols:
        columns[name]=eval(code, globals(), columns)

    if norm is None:
      w=self.data['p']
    else:
      w=eval(norm+'*p', globals(), columns)

    if fltr is None:
      I, x=histogram(columns[col], bins=bins, weights=w)
      if errors:
        N, _ = histogram(columns[col], bins=bins)
        dI = I/sqrt(maximum(N, 1))
    else:
      if isinstance(fltr, str):
        fltr=eval(fltr, globals(), columns)
      I, x=histogram(columns[col][fltr], bins=bins, weights=w[fltr])
      if errors:
        N, _ = histogram(columns[col][fltr], bins=bins)
        dI = I/sqrt(maximum(N, 1))

    if errors:
      return x, I, dI
    else:
      return x, I

  def project2d(self, xcol, ycol, bins=50, fltr=None, newcols=None):
    ''' 
      Generate 2D binned data for arbitrary binning and columns.
      
      The fltr argument can be a string with a filtering condition on the available columns.
      Syntax for this filtering follows numpy array convetions like (x>5)&(abs(L)<0.3) would
      be a valid statement.
      
      newcols can be a list of (name, code) tuples, that calculate new columns from
      existing ones, like newcols=[('q', '4*pi/L*sin(theta)')].
    '''
    columns=dict([(coli, self.data[coli]) for coli in self.data.dtype.names])
    if newcols is not None:
      for name, code in newcols:
        columns[name]=eval(code, globals(), columns)

    if fltr is None:
      I, y, x=histogram2d(columns[ycol], columns[xcol],
                        bins=bins, weights=self.data['p'])
    else:
      if isinstance(fltr, str):
        fltr=eval(fltr, globals(), columns)
      I, y, x=histogram2d(columns[ycol][fltr], columns[xcol][fltr],
                        bins=bins, weights=columns['p'][fltr])
    return x, y, I

  def plot(self, xcol='x', ycol='y', log=False, ax=None, bins=50, fltr=None, newcols=None,
           **kwds):
    if ax is None:
      import pylab
      ax=pylab.gca()

    x, y, I=self.project2d(xcol, ycol, bins=bins, fltr=fltr, newcols=newcols)

    if log:
      ax.pcolormesh(x, y, I, norm=LogNorm(), **kwds)
    else:
      ax.pcolormesh(x, y, I, **kwds)

    col_names=self.info['title'].split()
    cols=self.info['variables'].split()

    try:
      ax.set_xlabel(xcol+'-'+col_names[cols.index(xcol)])
    except ValueError:
      ax.set_xlabel(xcol)
    try:
      ax.set_ylabel(ycol+'-'+col_names[cols.index(ycol)])
    except ValueError:
      ax.set_ylabel(ycol)
    ax.set_title(self.info['component'])


  def plot1d(self, col='x', log=False, ax=None, bins=50, fltr=None, newcols=None,
             **kwds):
    if ax is None:
      import pylab
      ax=pylab.gca()

    x, I=self.project1d(col, bins=bins, fltr=fltr, newcols=newcols)

    if log:
      ax.semilogy((x[:-1]+x[1:])/2., I, **kwds)
    else:
      ax.plot((x[:-1]+x[1:])/2., I, **kwds)
    col_names=self.info['title'].split()
    cols=self.info['variables'].split()

    try:
      ax.set_xlabel(col+'-'+col_names[cols.index(col)])
    except ValueError:
      ax.set_xlabel(col)
    ax.set_ylabel('Intensity')
    ax.set_title(self.info['component'])

