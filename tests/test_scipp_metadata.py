import os
import tempfile
import numpy as np
import scipp as sc
import h5py
import json
import pytest
from unittest.mock import MagicMock
from mcstas_gisans.input_output import save_simulation_results_as_scipp

def test_save_simulation_results_metadata():
    """
    Dedicated test to verify that the sc.DataGroup metadata layout 
    not only exists but contains the exact expected values.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        savename = os.path.join(tmpdir, "test_output")

        # Mock Instrument
        inst = MagicMock()
        inst.is_tof_instrument = False
        inst.detector = MagicMock()
        inst.detector.direct_beam_centre_offset_x_nexus = 0.05
        inst.detector.direct_beam_centre_offset_y_nexus = -0.02
        inst.detector.sample_orientation = "vertical"
        inst.detector.pixels_x_nexus = 2
        inst.detector.pixels_y_nexus = 2
        inst.detector.get_pixel_positions.return_value = np.zeros((4, 3))
        
        inst.alpha_inc = np.deg2rad(0.5)
        inst.beam_angle = 1.2
        inst.nominal_source_sample_distance = 15.0
        inst.sample_detector_distance = 10.0
        inst.wavelength_selected = 6.0
        
        mock_scipp_da = sc.DataArray(
            data=sc.array(dims=['detector_id'], values=np.zeros(4)),
            coords={'detector_id': sc.array(dims=['detector_id'], values=np.arange(4))}
        )
        inst.create_scipp_container.return_value = mock_scipp_da
        
        # Mock Sample
        sample_mock = MagicMock()
        mock_module = MagicMock()
        mock_module.__file__ = __file__ # Use this test file as dummy source code
        sample_mock.get_module.return_value = mock_module
        
        params = {
            'instrument': inst,
            'instrument_name': 'test_instr',
            'sample': sample_mock
        }
        
        # Mock result (flattened 2D pixel hits)
        result = {
            'pixelHist': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'pixelHistWeightsSquared': np.array([[0.1, 0.2], [0.3, 0.4]])
        }
        
        # Mock parsed CLI args
        from argparse import Namespace
        args = Namespace()
        args.sample = "test_sample_name"
        args.sample_args = {"param1": 10}
        args.filename = "test_mcpl.mcpl"
        
        # Mock MCPL properties extracted from reader
        mcpl_metadata = {
            'sourcename': 'test_source_name',
            'nparticles': 12345,
            'comments': ['comment1', 'comment2']
        }
        
        # Call the actual save function
        save_simulation_results_as_scipp(savename, params, result, args, mcpl_metadata=mcpl_metadata)
        
        # Ensure the file was generated
        h5_path = savename + ".h5"
        assert os.path.exists(h5_path), "The HDF5 file was not created"
        
        # Load the file back using Scipp
        dataset = sc.io.hdf5.load_hdf5(h5_path)
        
        # 1. Assert structure
        assert isinstance(dataset, sc.DataGroup)
        assert 'data' in dataset
        assert 'instrument' in dataset
        assert 'sample' in dataset
        assert 'provenance' in dataset
        assert 'mcpl' in dataset
        
        # 2. Verify instrument metadata values
        inst_meta = dataset['instrument']
        assert inst_meta['name'].value == 'test_instr'
        assert inst_meta['is_tof_instrument'].value == False
        assert inst_meta['detector_centre_offset_x'].value == 0.05
        assert inst_meta['detector_centre_offset_y'].value == -0.02
        assert np.isclose(inst_meta['alpha_inc_deg'].value, 0.5)
        assert inst_meta['beam_angle'].value == 1.2
        assert inst_meta['sample_orientation'].value == "vertical"
        assert np.array_equal(inst_meta['source_position'].value, [0, 0, -15.0])
        assert inst_meta['wavelength_selected'].value == 6.0
        
        # 3. Verify sample metadata values
        sample_meta = dataset['sample']
        assert sample_meta['name'].value == 'test_sample_name'
        assert json.loads(sample_meta['arguments_json'].value) == {"param1": 10}
        assert len(sample_meta['script_content'].value) > 0
        assert "def test_save_simulation_results_metadata():" in sample_meta['script_content'].value
        
        # 4. Verify MCPL metadata values
        mcpl_meta = dataset['mcpl']
        assert mcpl_meta['filename'].value == 'test_mcpl.mcpl'
        assert mcpl_meta['sourcename'].value == 'test_source_name'
        assert mcpl_meta['nparticles'].value == 12345
        assert mcpl_meta['comments'].value == "comment1\ncomment2"
        
        # 5. Verify provenance metadata values
        prov_meta = dataset['provenance']
        assert 'cli_command' in prov_meta
        assert 'bornagain_version' in prov_meta
        assert 'mcstas_gisans_version' in prov_meta
        assert 'timestamp' in prov_meta
        
        args_dict = json.loads(prov_meta['cli_args_json'].value)
        assert args_dict['sample'] == "test_sample_name"
        
        # 6. Verify core data integration
        data = dataset['data']
        assert np.array_equal(data.values, [1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(data.variances, [0.1, 0.2, 0.3, 0.4])


def test_save_simulation_results_tof_metadata_and_variance():
    """
    TOF-instrument counterpart to test_save_simulation_results_metadata: the
    non-TOF path above was the only one covered before. This exercises the
    event-mode branch of save_simulation_results_as_scipp (reading the
    temporary per-worker HDF5 event buffer, binning into detector pixels,
    and attaching pixel positions), and specifically pins down the
    event-level variance convention (variance = weight**2, matching the
    non-TOF path's pixelHistWeightsSquared and what plot.py's
    scipp_binned.bins.sum() expects downstream).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        savename = os.path.join(tmpdir, "test_tof_output")

        # Build a fake per-worker TOF event buffer, matching what
        # run.py's _update_tof_buffer writes: 3 events landing in 2 pixels.
        detector_ids = np.array([0, 1, 0], dtype=np.int32)
        tofs = np.array([100.0, 200.0, 150.0], dtype=np.float32)
        weights = np.array([2.0, 3.0, 4.0], dtype=np.float64)

        temp_h5_path = os.path.join(tmpdir, "worker_events.h5")
        with h5py.File(temp_h5_path, 'w') as f:
            f.create_dataset('detector_id', data=detector_ids)
            f.create_dataset('tof', data=tofs)
            f.create_dataset('weight', data=weights)

        # Mock Instrument (TOF instrument, 1x2 pixel detector)
        inst = MagicMock()
        inst.is_tof_instrument = True
        inst.detector = MagicMock()
        inst.detector.direct_beam_centre_offset_x_nexus = 0.0
        inst.detector.direct_beam_centre_offset_y_nexus = 0.0
        inst.detector.sample_orientation = "horizontal"
        inst.detector.pixels_x_nexus = 1
        inst.detector.pixels_y_nexus = 2
        pixel_positions = np.array([[0.0, 0.0, 5.0], [0.0, 0.1, 5.0]])
        inst.detector.get_pixel_positions.return_value = pixel_positions

        inst.alpha_inc = np.deg2rad(0.5)
        inst.beam_angle = 1.2
        inst.nominal_source_sample_distance = 15.0
        inst.sample_detector_distance = 5.0

        sample_mock = MagicMock()
        mock_module = MagicMock()
        mock_module.__file__ = __file__
        sample_mock.get_module.return_value = mock_module

        params = {
            'instrument': inst,
            'instrument_name': 'test_tof_instr',
            'sample': sample_mock
        }

        result = {'temp_h5_path': temp_h5_path}

        from argparse import Namespace
        args = Namespace()
        args.model = "test_sample_name"
        args.sample_arguments = "radius=51"
        args.filename = "test_mcpl.mcpl"

        save_simulation_results_as_scipp(savename, params, result, args, mcpl_metadata=None)

        h5_path = savename + ".h5"
        assert os.path.exists(h5_path), "The HDF5 file was not created"
        # The temp per-worker event file must be cleaned up after consumption
        assert not os.path.exists(temp_h5_path), "Temporary TOF event file was not removed"

        dataset = sc.io.hdf5.load_hdf5(h5_path)

        assert isinstance(dataset, sc.DataGroup)
        assert 'data' in dataset
        assert 'mcpl' not in dataset, "No mcpl_metadata was given, so no mcpl group should be written"

        data = dataset['data']
        assert data.bins is not None, "TOF output should be binned event data"
        assert data.sizes['detector_id'] == 2
        assert np.array_equal(data.coords['position'].values, pixel_positions)

        summed = data.bins.sum()
        # Pixel 0 got events with weights 2.0 and 4.0; pixel 1 got weight 3.0.
        assert np.allclose(summed.values, [6.0, 3.0])
        # Variance must be sum(weight**2), not sum(weight): 2**2+4**2=20, 3**2=9.
        assert np.allclose(summed.variances, [20.0, 9.0])
