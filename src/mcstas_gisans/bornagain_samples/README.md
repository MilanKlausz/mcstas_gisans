# Built-in sample models and BornAgain versions

BornAgain changes its Python API between major versions: 22 replaced `ba.MultiLayer` and the layer roughness, and 24
replaced the materials and particle layouts. A sample model therefore works with some BornAgain versions only. The
built-in models are arranged by the BornAgain API they are written for:

```
bornagain_samples/
    ba21/   models for BornAgain 21
    ba22/   models for BornAgain 22 and 23
    ba24/   models for BornAgain 24
    *.py    models used with any BornAgain version (e.g. your own models; not version-checked)
```

A folder is named after the **first** BornAgain version of its API, not after every version it supports. There is
no `ba23/`, because BornAgain 23 uses the same API as 22: the models in `ba22/` are the ones for BornAgain 23.

## Which file is used

Each model file can state the BornAgain major versions it was tested with:

```python
BORNAGAIN_VERSIONS = (22, 23)   # tested with BornAgain 22 to 23
```

`--model <name>` picks, among the files called `<name>.py` in the version folders:

1. the file whose `BORNAGAIN_VERSIONS` contains the installed BornAgain version;
2. otherwise a file **without** `BORNAGAIN_VERSIONS`: it is treated as compatible with any version (it may still
   fail, e.g. with an error from BornAgain);
3. otherwise the run stops with an error that lists the versions the model exists for. A file is never used
   silently with a version outside its declared range: a newer BornAgain can break a model, or change its results
   without any error.

`--allow_untested_bornagain_version` overrides step 3: the file for the newest older version is used, with a
warning. `mg_run --help` lists the built-in models with the versions they are tested with.

Examples (`silica_100nm_air` exists in `ba21/`, `ba22/` and `ba24/`; `silica_air` in `ba21/` and `ba22/`):

| installed BornAgain | `silica_100nm_air` | `silica_air` |
|---|---|---|
| 21 | `ba21/` | `ba21/` |
| 23 | `ba22/` | `ba22/` |
| 24 | `ba24/` | error (or `ba22/` with `--allow_untested_bornagain_version`) |
| 25 | error (or `ba24/` with the flag) | error (or `ba22/` with the flag) |

## Adding a model

- Put the file into the folder of the BornAgain API it is written for (`ba22/` for BornAgain 22 and 23, `ba24/` for
  24), or directly into `bornagain_samples/` while trying it out.
- Stating `BORNAGAIN_VERSIONS` is suggested, but not required. Without it the model is used with every BornAgain
  version, which is convenient, but a future BornAgain may break it or change its results without notice.
- An implementation of an existing model for another BornAgain API keeps the same file name.
- For a model with declared versions, store its reference results with
  `python tests/make_builtin_model_references.py` (run it with a BornAgain version of the model's range).
- When a new BornAgain version comes out, run the tests with it, and extend `BORNAGAIN_VERSIONS` of the models that
  still pass. A model that needs the new API gets a new file in a new folder, e.g. `ba25/`. Models that are not
  updated are not available with the new version.

See the "Creating Custom Sample Models" page of the documentation for the API differences between the versions.
