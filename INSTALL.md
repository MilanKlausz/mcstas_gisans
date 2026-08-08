Standard installation
=====================

The recommended way to install `mcstas_gisans` is through Conda, using the provided `conda.yml` environment file.

Instead of Conda, `mcstas_gisans` can also be installed using `pip` as described in [# Alternative Installation with requirements.txt](#alternative-installation-with-requirementstxt).

> [!NOTE]
> **BornAgain Versioning & `conda.yml`**
> By default, **BornAgain 21.2** (and previous versions) is written in the `conda.yml` file because it is the last universal release distributed as pre-built wheels on PyPI for all operating systems.
>
> If you wish to use a more recent version of BornAgain (e.g. v22+, v23+, v24+), you can edit the `conda.yml` file before creating the environment.
>
> **Warning (macOS Users):** In order to install recent BornAgain versions (v22+) on macOS, pre-built PyPI wheels are not available, so extra build steps are required as described in [# Installing Recent BornAgain Versions (macOS Build Guide)](#installing-recent-bornagain-versions-macos-build-guide).

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

The required Python packages can be installed using the `requirements.txt` file:
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

# Installing Recent BornAgain Versions (macOS Build Guide)

To install BornAgain versions newer than 21.2 (such as v22, v23, v24, or v25) on macOS:

1. **Build the Python wheel:** Follow the official BornAgain build-from-source instructions for Unix systems up to the step that creates the Python wheel file (`ninja ba_wheel` or `make ba_wheel`).

2. **Locate the generated `.whl` file:**
   ```bash
   find <build_directory> -name "*.whl"
   ```

3. **Edit `conda.yml` to point to the local wheel:**
   In `conda.yml`, locate the `- pip:` block. Replace the default `- bornagain==21.2` line with the path to your local wheel file, for example:
   ```yaml
     - pip:
       # - bornagain==21.2
       - ./bornagain_versions/ba24/bornagain-24.1-cp311-cp311-macosx_11_0_x86_64.whl
   ```

4. **Create and activate the environment:**
   ```bash
   conda env create -f conda.yml
   conda activate mcstas_gisans
   ```