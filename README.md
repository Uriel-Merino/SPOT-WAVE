# Spot Wave

Library that systematizes the `SPOT_WAVE.ipynb` pipeline: time-frequency
analysis with [wavepal](https://github.com/guillaumelenoir/WAVEPAL) +
CARMCMC, wavelet filtering (single or double) of stellar activity in
radial velocities, and orbital fitting with
[CONAN](https://github.com/mlendl42/CONAN3) on the winning residuals.

**Scope of this version:** a single target per run. For the systematic
multi-target sweep (many files / many systems in parallel) keep using
`analyze_filter_sweep_v3.py` + `analyze_filter_sweep_syst.py` from the
original pipeline — both could be rewritten in the future to build on
this library instead of duplicating the logic.

## Structure

```
spot_wave/
├── __init__.py      # public API of the package
├── __main__.py       # CLI: python -m spot_wave single|double ...
├── __version__.py
├── utils.py          # setup_carmcmc, load_rv_file, test_cwt_feasibility, wavepal_analyze
├── filter.py          # single_filter_sweep (1 band) and run_double_filter_sweep (2 bands, order 1/2)
├── conan_fit.py        # run_conan_fit, extract_conan_metrics, extract_k_posterior
└── scalogram.py        # analyze_and_plot, w0_loop (sections 2 and 2.1 of the notebook)
```

## Installation

External dependencies:

- **CONAN** — your fork: [Uriel-Merino/CONAN](https://github.com/Uriel-Merino/CONAN).
  This one installs **automatically** when installing `spot_wave`,
  because the fork keeps CONAN's modern `pyproject.toml`/`setup.py`
  (declared in `pyproject.toml` as
  `conan-exoplanet @ git+https://github.com/Uriel-Merino/CONAN.git`).

- **wavepal + CARMCMC** — your fork:
  [Uriel-Merino/WAVEPAL](https://github.com/Uriel-Merino/WAVEPAL) (forked
  from `guillaumelenoir/WAVEPAL`). This one does **NOT** install itself:
  it's a package with Python 2 origins (you already ported the
  `wavepal/` folder to Python3) whose installation goes through
  `Linux_install.sh` / `MacOSX_install.sh`, because the `carmcmc/`
  subfolder (CARMCMC/`carma_pack`) needs to be compiled against
  **BOOST** and **ARMADILLO** (system C++ libraries) — something pip
  cannot resolve on its own. Install it manually:

  ```bash
  git clone https://github.com/Uriel-Merino/WAVEPAL.git
  cd WAVEPAL
  sh Linux_install.sh   # or MacOSX_install.sh on Mac
  ```

  then point `SPOT_WAVE_CARMCMC_PATH` to
  `<clone_path>/WAVEPAL/carmcmc/carma_pack/src` (or pass it to
  `setup_carmcmc(path=...)`).

- **gls** (Zechmeister & Kürster periodogram) — the `gls.py` module you
  already use in your pipeline; add it to the PYTHONPATH or copy it into
  the repo.

```bash
git clone <your-spot_wave-repo>
cd spot_wave
pip install -e .          # installs spot_wave + CONAN automatically

# wavepal + CARMCMC, manually (see above):
git clone https://github.com/Uriel-Merino/WAVEPAL.git
cd WAVEPAL && sh Linux_install.sh && cd ..

export SPOT_WAVE_CARMCMC_PATH=$(pwd)/WAVEPAL/carmcmc/carma_pack/src
export PYTHONPATH=$PYTHONPATH:/path/to/gls.py
```

## Usage — Python API

```python
import spot_wave as sw

sw.setup_carmcmc()  # optional if CARMCMC is already on the PYTHONPATH
t, rv, rv_err, instruments = sw.load_rv_file("my_target.dat")

# --- Mode 1: simple filter (one band, one w0 range) ---
w0_grid = sw.make_w0_grid(5.5, 20.0, 0.5)
best = sw.single_filter_sweep(
    t, rv, rv_err, w0_grid, band=(54.5, 55.5),
    p_rot=55.0, p_rot_half=27.5, p_planeta=19.25,
)

# --- Mode 2: double filter (two bands: Prot and Prot/2, order 1 and/or 2) ---
order1 = dict(
    w0_grid_1=sw.make_w0_grid(5.5, 7.0, 0.5),    # filters the Prot band first
    w0_grid_2=sw.make_w0_grid(14.5, 20.0, 0.25), # then the Prot/2 band
    hw_1=0.5, hw_2=0.5,
)
order2 = dict(
    w0_grid_1=sw.make_w0_grid(11.5, 14.5, 0.1),  # filters the Prot/2 band first
    w0_grid_2=sw.make_w0_grid(5.5, 7.0, 0.1),    # then the Prot band
    hw_1=0.5, hw_2=0.5,
)
best = sw.run_double_filter_sweep(
    t, rv, rv_err, prot_value=55.0, prot_half_value=27.5, planet_period=19.25,
    order1=order1, order2=order2,
)

sw.save_winner_file(t, best["residuals"], rv_err, "winner_residuals.dat")

# --- CONAN on the winning residual ---
planet_pars = dict(
    T_0=[(3.2, 0.2)], Period=[(19.25, 0.01)],
    Eccentricity=[0], omega=[90], K=[(0.0, 5.0, 10.0)],
)
sw.run_conan_fit(
    filtered_data_files=["winner_residuals.dat"], data_path=".",
    output_folder="conan_out/", planet_pars=planet_pars,
    m_star=(0.467, 0.02), gamma_prior=[(0, 30)],
)
metrics = sw.extract_conan_metrics("conan_out/", n_data_points=len(t))
k_med, k_lo1, k_hi1, k_lo3, k_hi3 = sw.extract_k_posterior("conan_out/")
```

## Usage — CLI

```bash
# Simple filter
python -m spot_wave single \
    --rv-file my_target.dat --out-dir results/ \
    --p-rot 55.0 --p-planeta 19.25 \
    --band-lo 54.5 --band-hi 55.5 \
    --w0-min 5.5 --w0-max 20.0 --w0-step 0.5

# Double filter
python -m spot_wave double \
    --rv-file my_target.dat --out-dir results/ \
    --p-rot 55.0 --p-planeta 19.25 \
    --w0-1-min 5.5 --w0-1-max 7.0 --w0-1-step 0.5 \
    --w0-2-min 14.5 --w0-2-max 20.0 --w0-2-step 0.25 \
    --hw-prot 0.5 --hw-half 0.5 --orders 1,2
```

## Notes / to do

- The empirical `S_score` (`eta_activity / eta_planeta` via GLS) is the
  same selection criterion as in `analyze_filter_sweep_v3.py`; it lives
  in `filter.compute_empirical_gls_score`.
- `scalogram.py` covers the "2. SCALOGRAM" and "2.1. Loop w0" sections of
  the notebook (including capturing the "Re-estimated period range" that
  wavepal prints).
- If you share your current `__init__.py` / `filter.py` / `utils.py`,
  this version can be merged with yours instead of replacing it.
- wavepal/CARMCMC and CONAN are your own forks
  ([WAVEPAL](https://github.com/Uriel-Merino/WAVEPAL),
  [CONAN](https://github.com/Uriel-Merino/CONAN)), not the original
  repos — any fix you made there (e.g. the "Fixed bug" commit in
  `carmcmc`/`test`) is already included if you install from those URLs.
