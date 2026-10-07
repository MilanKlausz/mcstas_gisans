Standard installation
=====================

The recommended way to install `mcstas_gisans` is through Conda, using the provided `conda.yml` environment file.

Instead of Conda, `mcstas_gisans` can also be installed using `pip` as described in [# Alternative Installation with requirements.txt](#alternative-installation-with-requirementstxt).

> [!NOTE]
> **BornAgain Versioning & `conda.yml`**
> By default, **BornAgain 23.0** is written in the `conda.yml` file (and BornAgain >= 23.0 in `requirements.txt` and `pyproject.toml`). This is the version `mcstas_gisans` is developed and tested with; older versions are not supported any more. BornAgain 24.1 also runs (a systematic benchmark is pending); to use it, edit the `conda.yml` file before creating the environment.
>
> **Warning (macOS Users):** PyPI provides pre-built BornAgain 23.0 (and 24.1) wheels only for Linux (x86_64, glibc >= 2.31); for macOS only the unsupported 21.x versions are available there. On macOS, BornAgain therefore has to be built from source into a local wheel, as described in [# Installing BornAgain on macOS (Build Guide)](#installing-bornagain-on-macos-build-guide), and the `- bornagain==23.0` line in `conda.yml` replaced with the path of that wheel (with `pip`, install the wheel before `requirements.txt`).

To create the environment using Conda, run:
```bash
conda env create -f conda.yml
```

Then activate the environment:
```bash
conda activate mcstas_gisans
```

---

# Alternative Installation with requirements.txt

Alternatively, `mcstas_gisans` can be installed with `pip`—preferably in a virtual environment created and activated by the commands:
```bash
python -m venv myenv
source myenv/bin/activate
```

The required Python packages can be installed using the `requirements.txt` file (on macOS, first install a locally built BornAgain wheel with `pip install <path to the .whl file>`, see [# Installing BornAgain on macOS (Build Guide)](#installing-bornagain-on-macos-build-guide)):
```bash
pip install -r requirements.txt
```

`mcstas_gisans` can then be installed with:
```bash
pip install .
```
or in editable mode for developers:
```bash
pip install -e .
```

---

# Installing BornAgain on macOS (Build Guide)

PyPI has no macOS wheels of BornAgain 23.0 or newer, so on macOS the default BornAgain 23.0 (or a newer version, e.g. 24.1) has to be built from source:

1. **Build the Python wheel:** Follow the official BornAgain build-from-source instructions for Unix systems up to the step that creates the Python wheel file (`ninja ba_wheel` or `make ba_wheel`).

2. **Locate the generated `.whl` file:**
   ```bash
   find <build_directory> -name "*.whl"
   ```

3. **Edit `conda.yml` to point to the local wheel:**
   In `conda.yml`, locate the `- pip:` block. Replace the default `- bornagain==23.0` line with the path to your local wheel file, for example:
   ```yaml
     - pip:
       # - bornagain==23.0
       - ./bornagain_versions/ba23/bornagain-23.0-cp311-cp311-macosx_11_0_x86_64.whl
   ```

4. **Create and activate the environment:**
   ```bash
   conda env create -f conda.yml
   conda activate mcstas_gisans
   ```