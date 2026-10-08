Standard installation
=====================

The recommended way to install `mcstas_gisans` is through Conda, using the provided `conda.yml` environment file.

Instead of Conda, `mcstas_gisans` can also be installed using `pip` as described in [# Alternative Installation with requirements.txt](#alternative-installation-with-requirementstxt).

> [!NOTE]
> **BornAgain Versioning & `conda.yml`**
> By default, **BornAgain 23.0** is written in the `conda.yml` file (and `bornagain>=22.0,<24` in `requirements.txt` and `pyproject.toml`). This is the version `mcstas_gisans` is developed and tested with; BornAgain 22 (22.2) passes the test suite as well, older versions are not supported any more. BornAgain 24 changed the Python API of materials and particle layouts; the built-in sample models have implementations for each BornAgain API in version folders (`bornagain_samples/ba21`, `ba22`, `ba24`, each file declaring the versions it is tested with; with BornAgain 24 `silica_100nm_air` and the two liquid-surface models are available). With `--use_avg_materials` (used by all fits so far) BornAgain 24.1 gives the same results as 23.0 (paper example and the built-in models); without it, particles inside a lower layer (e.g. the liquid-surface models) give a different, much lower intensity with 24.1, and two specular tests fail with 24.1 (0.1% difference). Hence the upper limit `<24` until this is understood.
>
> **Warning (macOS Users):** PyPI provides pre-built BornAgain 23.0 wheels for Linux (x86_64, glibc >= 2.31; 24.1: glibc >= 2.35) and Windows (x86_64); for macOS only the unsupported 21.x versions are available there. On macOS, BornAgain therefore has to be built from source into a local wheel, as described in [# Installing BornAgain on macOS (Build Guide)](#installing-bornagain-on-macos-build-guide), and the `- bornagain==23.0` line in `conda.yml` replaced with the path of that wheel (with `pip`, install the wheel before `requirements.txt`).

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

PyPI has no macOS wheels of BornAgain 23.0 or newer, so on macOS the default BornAgain 23.0 has to be built from source. The official BornAgain instructions for building from source on Unix systems (https://bornagainproject.org, "Build and install") describe the dependencies; in short:

1. **Build and install the two BornAgain libraries, then BornAgain** (each with CMake and Ninja, installing into one prefix directory and passing it to the next build with `-DCMAKE_PREFIX_PATH=<prefix>`). The library versions must match the BornAgain version:

   | BornAgain | libheinz | libformfactor |
   |---|---|---|
   | 23.0 | v2.0.1 | v0.3.2 |
   | 24.1 | v4.1.0 | v0.4.0 |

   Configure BornAgain with the Python interpreter of the target environment (`-DPython3_EXECUTABLE=<python 3.11>`), then build the wheel with `ninja ba_wheel`.

2. **Locate the generated `.whl` file** (in `<build_directory>/py/wheel/`):
   ```bash
   find <build_directory> -name "*.whl"
   ```

3. **Make the wheel self-contained.** On macOS the wheel made by `ba_wheel` does not include the external libraries (libformfactor, GSL, FFTW, libcerf, ...) and, for 24.1, not even all of BornAgain's own libraries: it loads them from the build and prefix directories by absolute path. If these directories are moved or deleted later, `import bornagain` fails (e.g. `libformfactor.*.dylib` not found). Either keep the build directory, the prefix directory and the dependency environment where they are, or bundle the libraries into the wheel with [delocate](https://github.com/matthew-brett/delocate):
   ```bash
   delocate-wheel --exclude libpython3.11.dylib --exclude libomp.dylib --exclude libc++.1 -w <output_dir> <wheel>
   ```
   Do not bundle `libomp`: a second copy next to the one of the Conda environment (used by SciPy) aborts with `OMP: Error #15`. For 24.1, the references between BornAgain's own modules (`@rpath/_libBornAgain*.24.1.so`) have to be redirected to the copies inside the wheel (`install_name_tool -change ... @loader_path/...`) before running delocate, otherwise both copies are loaded.

4. **Edit `conda.yml` to point to the local wheel:**
   In `conda.yml`, locate the `- pip:` block. Replace the default `- bornagain==23.0` line with the path to your local wheel file (keep the wheel outside the repository), for example:
   ```yaml
     - pip:
       # - bornagain==23.0
       - /path/to/wheels/bornagain-23.0-cp311-cp311-macosx_11_0_x86_64.whl
   ```

5. **Create and activate the environment:**
   ```bash
   conda env create -f conda.yml
   conda activate mcstas_gisans
   ```