import pytest
from mcstas_gisans.instrument_defaults import reset_instrument_defaults

@pytest.fixture(autouse=True)
def auto_reset_instrument_defaults():
    reset_instrument_defaults()
    yield
    reset_instrument_defaults()
