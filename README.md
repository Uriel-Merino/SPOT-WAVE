# Spot Wave

Librería que sistematiza el pipeline de `SPOT_WAVE.ipynb`: análisis
tiempo-frecuencia con [wavepal](https://github.com/guillaumelenoir/WAVEPAL) +
CARMCMC, filtrado wavelet (simple o doble) de la actividad estelar en
velocidades radiales, y ajuste orbital con [CONAN](https://github.com/mlendl42/CONAN3)
sobre los residuos ganadores.

**Alcance de esta versión:** un único target por ejecución. Para el barrido
sistemático multi-target (muchos ficheros / muchos sistemas en paralelo)
sigue usando `analyze_filter_sweep_v3.py` + `analyze_filter_sweep_syst.py`
del pipeline original — ambos podrían reescribirse en el futuro para
apoyarse en esta librería en vez de duplicar la lógica.

## Estructura

```
spot_wave/
├── __init__.py      # API publica del paquete
├── __main__.py       # CLI: python -m spot_wave single|double ...
├── __version__.py
├── utils.py          # setup_carmcmc, load_rv_file, test_cwt_feasibility, wavepal_analyze
├── filter.py          # single_filter_sweep (1 banda) y run_double_filter_sweep (2 bandas, orden 1/2)
├── conan_fit.py        # run_conan_fit, extract_conan_metrics, extract_k_posterior
└── scalogram.py        # analyze_and_plot, w0_loop (secciones 2 y 2.1 del notebook)
```

## Instalación

Dependencias externas:

- **CONAN** — tu fork: [Uriel-Merino/CONAN](https://github.com/Uriel-Merino/CONAN).
  Este sí se instala **automáticamente** al instalar `spot_wave`, porque el
  fork conserva el `pyproject.toml`/`setup.py` modernos de CONAN (declarado
  en `pyproject.toml` como `conan-exoplanet @ git+https://github.com/Uriel-Merino/CONAN.git`).

- **wavepal + CARMCMC** — tu fork:
  [Uriel-Merino/WAVEPAL](https://github.com/Uriel-Merino/WAVEPAL) (forked
  de `guillaumelenoir/WAVEPAL`). Este **NO** se instala solo: es un
  paquete originalmente en Python 2 (tú ya portaste la carpeta `wavepal/`
  a Python3) cuya instalación pasa por `Linux_install.sh` /
  `MacOSX_install.sh`, porque la subcarpeta `carmcmc/` (CARMCMC/
  `carma_pack`) necesita compilarse contra **BOOST** y **ARMADILLO**
  (librerías C++ del sistema) — algo que pip no puede resolver por su
  cuenta. Instálalo a mano:

  ```bash
  git clone https://github.com/Uriel-Merino/WAVEPAL.git
  cd WAVEPAL
  sh Linux_install.sh   # o MacOSX_install.sh en Mac
  ```

  y luego apunta `SPOT_WAVE_CARMCMC_PATH` a
  `<ruta_al_clone>/WAVEPAL/carmcmc/carma_pack/src` (o pásala a
  `setup_carmcmc(path=...)`).

- **gls** (periodograma Zechmeister & Kürster) — el módulo `gls.py` que ya
  usas en tu pipeline; añádelo al PYTHONPATH o cópialo dentro del repo.

```bash
git clone <tu-repo-spot_wave>
cd spot_wave
pip install -e .          # instala spot_wave + CONAN automaticamente

# wavepal + CARMCMC, a mano (ver arriba):
git clone https://github.com/Uriel-Merino/WAVEPAL.git
cd WAVEPAL && sh Linux_install.sh && cd ..

export SPOT_WAVE_CARMCMC_PATH=$(pwd)/WAVEPAL/carmcmc/carma_pack/src
export PYTHONPATH=$PYTHONPATH:/ruta/donde/esta/gls.py
```

## Uso — API de Python

```python
import spot_wave as sw

sw.setup_carmcmc()  # opcional si CARMCMC ya esta en el PYTHONPATH
t, rv, rv_err, instruments = sw.load_rv_file("mi_target.dat")

# --- Modo 1: filtro simple (una banda, un rango de w0) ---
w0_grid = sw.make_w0_grid(5.5, 20.0, 0.5)
best = sw.single_filter_sweep(
    t, rv, rv_err, w0_grid, band=(54.5, 55.5),
    p_rot=55.0, p_rot_half=27.5, p_planeta=19.25,
)

# --- Modo 2: filtro doble (dos bandas: Prot y Prot/2, orden 1 y/o 2) ---
order1 = dict(
    w0_grid_1=sw.make_w0_grid(5.5, 7.0, 0.5),    # filtra banda Prot primero
    w0_grid_2=sw.make_w0_grid(14.5, 20.0, 0.25), # luego banda Prot/2
    hw_1=0.5, hw_2=0.5,
)
order2 = dict(
    w0_grid_1=sw.make_w0_grid(11.5, 14.5, 0.1),  # filtra banda Prot/2 primero
    w0_grid_2=sw.make_w0_grid(5.5, 7.0, 0.1),    # luego banda Prot
    hw_1=0.5, hw_2=0.5,
)
best = sw.run_double_filter_sweep(
    t, rv, rv_err, prot_value=55.0, prot_half_value=27.5, planet_period=19.25,
    order1=order1, order2=order2,
)

sw.save_winner_file(t, best["residuals"], rv_err, "residuos_ganador.dat")

# --- CONAN sobre el residuo ganador ---
planet_pars = dict(
    T_0=[(3.2, 0.2)], Period=[(19.25, 0.01)],
    Eccentricity=[0], omega=[90], K=[(0.0, 5.0, 10.0)],
)
sw.run_conan_fit(
    filtered_data_files=["residuos_ganador.dat"], data_path=".",
    output_folder="conan_out/", planet_pars=planet_pars,
    m_star=(0.467, 0.02), gamma_prior=[(0, 30)],
)
metrics = sw.extract_conan_metrics("conan_out/", n_data_points=len(t))
k_med, k_lo1, k_hi1, k_lo3, k_hi3 = sw.extract_k_posterior("conan_out/")
```

## Uso — CLI

```bash
# Filtro simple
python -m spot_wave single \
    --rv-file mi_target.dat --out-dir resultados/ \
    --p-rot 55.0 --p-planeta 19.25 \
    --band-lo 54.5 --band-hi 55.5 \
    --w0-min 5.5 --w0-max 20.0 --w0-step 0.5

# Filtro doble
python -m spot_wave double \
    --rv-file mi_target.dat --out-dir resultados/ \
    --p-rot 55.0 --p-planeta 19.25 \
    --w0-1-min 5.5 --w0-1-max 7.0 --w0-1-step 0.5 \
    --w0-2-min 14.5 --w0-2-max 20.0 --w0-2-step 0.25 \
    --hw-prot 0.5 --hw-half 0.5 --orders 1,2
```

## Notas / pendiente

- El `S_score` empírico (`eta_actividad / eta_planeta` vía GLS) es el mismo
  criterio de selección que en `analyze_filter_sweep_v3.py`; vive en
  `filter.compute_empirical_gls_score`.
- `scalogram.py` cubre las secciones "2. SCALOGRAM" y "2.1. Loop w0" del
  notebook (incluida la captura del "Re-estimated period range" que
  imprime wavepal).
- Si compartes tus `__init__.py` / `filter.py` / `utils.py` actuales, se
  puede fusionar esta versión con la tuya en vez de sustituirla.
- wavepal/CARMCMC y CONAN son tus propios forks
  ([WAVEPAL](https://github.com/Uriel-Merino/WAVEPAL),
  [CONAN](https://github.com/Uriel-Merino/CONAN)), no los repos originales
  — cualquier fix que hayas metido ahí (p.ej. el commit "Fixed bug" en
  `carmcmc`/`test`) ya viene incluido si instalas desde esas URLs.
