Standard installation
=====================

The recommended way to install mcstas_gisans is through Conda, using the 
provided conda.yml environment file by running the command:
```
conda env create -f conda.yml
```
Then the environment can be activated by running:
```
conda activate mcstas_gisans
```

# Alternative Installation

Alternatively, mcstas_gisans can be installed with pip - preferably in a
virtual environment created and activated by the commands:
```
python -m venv myenv
source myenv/bin/activate
```
The required python packages can be installed using the requirements.txt file
with the command:
```
pip install -r requirements.txt
```
mcstas_gisans can then be installed with the command:
```
pip install .
```
or in editable mode for developers with the command:
```
pip install -e .
```

# Warning about the BornAgain version
mcstas_gisans has been developed using BornAgain version 21.1. Both the
conda.yml and requirements.txt files define this version. Newer versions might
work but they are not tested.