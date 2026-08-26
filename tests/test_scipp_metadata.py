import os
import tempfile
import numpy as np
import scipp as sc
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
