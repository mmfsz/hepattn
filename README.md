# hepattn

We present a general end-to-end ML approach for particle physics reconstruction by adapting cutting-edge object detection techniques.
Our work demonstrates that a single encoder-decoder transformer can solve many different reconstruction problems that traditionally required specialised, task-specific approaches.

Our method has been successfully applied to various reconstruction tasks and detector setups:

- **Pixel cluster splitting** - ATLAS [[PUB][tide]]
- **Hit filtering** - TrackML [[arXiv][trackml]], ITk [WIP]
- **Tracking** - TrackML [[arXiv][trackml]], ATLAS [[PUB][tide]]
- **Primary vertexing** - *Interested in working on this? Get in touch!*
- **Secondary vertexing** - Delphes [[EPJC][vertexing]]
- **Particle flow** - CLIC [[arXiv][glow]]
- **End-to-end reconstruction** - CLD [[ML4Jets][ml4jets]]
- **Muon Tracking** - ATLAS [[ConnectingTheDots][ctd]]

[tide]: https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2025-045/
[trackml]: https://arxiv.org/abs/2411.07149
[vertexing]: https://link.springer.com/article/10.1140/epjc/s10052-024-13374-5
[glow]: https://arxiv.org/abs/2508.20092
[ml4jets]: https://indico.cern.ch/event/1526677/contributions/6530938/
[ctd]: https://indico.cern.ch/event/1499357/contributions/6621917/

## ✨ Key Features

- **🏗️ Modular architecture**: Encoder, decoder, and task modules for flexible experimentation
- **⚡ Efficient attention**: Seamlessly switch between torch SDPA, FlashAttention, and FlexAttention
- **🔬 Cutting-edge transformers**: HybridNorm, LayerScale, value residuals, register tokens, local attention
- **🚀 Performance optimised**: Full `torch.compile` and nested tensor support
- **🧪 Thoroughly tested**: Comprehensive tests across multiple reconstruction tasks
- **📦 Easy deployment**: Packaged with Pixi for reproducible environments


## 🛠️ Setup

First clone the repository:

```shell
git clone git@github.com:samvanstroud/hepattn.git
cd hepattn
```

We recommend using a container to set up and run the code.
This is necessary if your system's `libc` version is `<2.28`
due to requirements of recent `torch` versions.
We use `pixi`'s CUDA image, which you can access with:

```shell
apptainer pull pixi.sif docker://ghcr.io/prefix-dev/pixi:0.54.1-jammy-cuda-12.8.1
apptainer shell --nv pixi.sif
```

**📝 Note**: If you are not using the `pixi` container, you will need to make sure
`pixi` is installed according to https://pixi.sh/latest/installation/.

You can then install the project with locked dependencies:

```shell
pixi install --locked
```

**📝 Note**: The `default` environment targets GPU machines and installs FA2.
The `clic` environment is `default` plus the CLIC analysis packages (`fastjet`,
`energyflow`, `vector`, `pathos`), with the same pinned torch and flash-attention
build, so it can both train and run the jet and substructure analysis; use it for
anything CLIC. See the [pyproject.toml](pyproject.toml) or
[setup/isambard.md](setup/isambard.md) for more information.

### The `lap1015` Extension

The `lap1015` linear assignment solver is a C++ extension vendored in
[`src/lap1015`](src/lap1015) and compiled from that source by `pixi install`. It
must be built from *this* repository's source, and not from an older or upstream
build, because only this version releases the GIL while solving. Without that, the
threaded matcher serialises: `Matcher(parallel_solver=True, n_jobs=16)` quietly runs
sixteen threads that all queue behind each other, and the CLIC default
`lap1015_late` solver ends up roughly 2x slower than `scipy`.

**Check your build in one line:**

```shell
pixi run python -c "import lap1015; print(lap1015.releases_gil)"
```

`True` is what you want, and `pixi reinstall hepattn` is how you get it.

`False` needs care: the flag is set at **compile** time, so it is `False` on any
extension that was patched in place rather than rebuilt — even one that does
release the GIL. It reports how the binary was produced, not how it behaves. A
`False` here means "this build is not reproducible from `src/lap1015`", which is
reason enough to reinstall, but it is not on its own evidence that matching is
serialised.

To find out whether the solve is *actually* threaded, time it. On a B200 node at
the CLIC geometry a serialised solve costs ~5.6 s per step against ~0.8 s with
`n_jobs=16`; anything near the latter is threaded, whatever the flag says. That
distinction cost a day of investigation once — the warning is about provenance,
the timing is about behaviour.

**To rebuild:**

```shell
pixi reinstall hepattn
```

This recompiles the extension from `src/lap1015/src/main.cpp`. A plain `pixi install`
will *not* do it if the environment already exists — pixi sees the package version
unchanged and skips it, which is how a build can sit stale for weeks across edits to
the C++ source.

Two things guard this, and are worth knowing about if you change the build:

- `strict-config = false` in [pyproject.toml](pyproject.toml) lets the build
  tolerate the `pixi-conda-environment` config-setting that pixi passes to the
  backend. Without it, scikit-build-core rejects the unknown option and
  `pixi install` fails outright on a fresh clone with
  `Unrecognized options in config-settings`.
- `tests/matching/test_solvers.py::test_lap1015_releases_gil` fails if the
  installed extension holds the GIL, so a stale build is caught by the test suite
  rather than by a warning nobody reads.

Note that scikit-build-core's `editable.rebuild = true` is *not* used, though it
looks like the obvious fix. Its import-time rebuild runs under the system `cmake`,
which cannot find the Python development headers, so it turns a stale extension
into an unimportable one. Rebuild explicitly instead.

## 🌟 Activating the Environment

To run the installed environment, use:

```shell
pixi shell
```

Multiple environments are configured in `pyproject.toml` for different hardware setups and experiments (`default`, `cpu`, `isambard`, `clic`, `tide`, `ci`). Use `-e <env>` to specify a specific environment.

You can close the environment with `exit`.
See the [`pixi shell` docs](https://pixi.sh/latest/reference/cli/pixi/shell/) for more information.

## 🧪 Running Tests

Once inside the environment, if a GPU and relevant external data are available, just run:

```shell
pytest
```

To test parts of the code that don't require a GPU, run:

```shell
pytest -m 'not gpu'
```

To test parts of the code that don't require external input data, run:

```shell
pytest -m 'not requiresdata'
```

The current CI only tests the parts of the code that don't require a GPU or external input data:

```shell
pytest -m 'not gpu and not requiresdata'
```

**📝 Note**: If you encounter import errors for missing modules like `numba` when running tests in the `default` environment, switch to the appropriate experiment environment or use the `ci` environment which includes all required dependencies for tests (e.g. `pixi run -e ci pytest -m 'not gpu and not requiresdata'`).


## 🏃 Run Experiments

See experiment directories for instructions on how to run experiments.

- [TrackML Tracking](src/hepattn/experiments/trackml/)
- [CLIC Particle Flow](src/hepattn/experiments/clic/)

## 📖 Terminology

To ensure clarity and consistency throughout this project, we use the following definitions:

- **constituent** - input entities that go into the encoder/decoder, e.g. inner detector hits
- **object** - reconstructed outputs from the decoder, e.g. reconstructed charged particle tracks
- **input** - (also `input_object`) generic term for any input to a module (could be constituents, objects, etc)
- **output** - generic term for any output from a module (could be objects, predictions, or intermediates)

## 🤝 Contributing

If you would like to contribute, please lint and format code with

```shell
ruff check --fix .
ruff format .
```

You can also set up pre-commit hooks to automatically run these checks before committing:

```shell
pre-commit install
```

## 📄 Citing

If you use this software in your research, please cite it using the citation information available in the GitHub repository sidebar (generated from [`CITATION.cff`](CITATION.cff)).
Please also cite [our papers](#hepattn) if they are relevant to your work.
