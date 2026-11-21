# gfp_prog_tools

General Python helper functions: data handling, confidence intervals, graph formatting.

```python
from gfp_prog_tools import Set_graph_params, Save_figure, Wrap_marker, Calculate_student_t_confidence_interval_curve_fit
```

## Install

On a computer where the repository is cloned to `%USERPROFILE%\gfp_prog_tools` (edits take effect immediately):

```
pip install -e %USERPROFILE%\gfp_prog_tools
```

Anywhere else, or as a dependency in a project's `requirements.txt`, pin a version tag:

```
gfp_prog_tools @ git+https://github.com/gfp96/gfp_prog_tools.git@v0.5.2
```

## Claude Code conventions

[claude/CLAUDE.md](claude/CLAUDE.md) holds the Python conventions Claude Code should follow.
Copy it to `%USERPROFILE%\.claude\CLAUDE.md` on each computer so it applies in every project
(append it if that file already exists).

## Versions

### v0.5.2

`soils`:

- New: `Sand.Get_sanisand04_G0(pe, Pref=101.325, calculate_ci=False, dpe=0, de=0, e=None)`, the SANISAND04
  (Dafalias & Manzari, 2004) elastic shear modulus G = g0_dm04 Pref (2.97 - e)^2/(1 + e) (p'/Pref)^0.5, in the
  unit of Pref. Raises ValueError if the class has no `g0_dm04` (new `Sand` defaults: `g0_dm04 = -1`, `dg0_dm04 = 0`).
- `Get_hardin_drnevich_G0`: new optional argument `e` (void ratio, float or array); None keeps using `s.e`.

### v0.5.1

`soils`: small strain shear modulus G0 in the unit of Pref. Breaking change for G0 values.

- `Get_hardin_drnevich_G0`: G0 = G0A Pref (G0B - e)^2/(1 + e) (p'/Pref)^G0C, and `Get_fitted_G0`:
  G0 = G0D Pref (p'/Pref)^G0E, now return G0 in the unit of Pref (kPa by default) instead of MPa.
  G0A and G0D are dimensionless; stored values rescaled by 1000/101.325 so that G0 is unchanged:
  NE34 G0A 132.98 -> 1312.41 (dG0A 4.27 -> 42.14), G0D 99.98 -> 986.73 (dG0D 0.26 -> 2.57);
  Toyoura G0A 95.5 -> 942.51; HostunRF G0A 21.46 -> 211.79 (dG0A 0.67 -> 6.61).
  Karlsruhe_fine_sand G0A = 150 was already dimensionless (DM04 g0): its G0 drops from ~10x too high to the intended value.
- `calculate_ci=True` of both G0 methods: works with array `pe`; `Sand` defines `dG0A`, `dG0B`, `dG0C` = 0,
  so soils without stored uncertainties no longer raise AttributeError.
- Docstrings: `Get_P2Psand_G0` returns the unit of Pref (not MPa); `Get_G0_menq` returns MPa.

### v0.5.0

`soils`: bulk modulus laws per loading phase. Breaking changes for `Sand` and subclasses.

- Each sand class holds one law per phase, `K_virgin_loading` and `K_unloading`:
  `{"model": ..., "params": {name: (value, half-width of the 0.95 ci)}, "source": ...}`.
  Equation forms (`BULK_MODULUS_MODELS`):
  - `"hardin_drnevich"`: K = A (B − e)²/(1 + e) (p′/Pref)^C (Sand default for virgin loading)
  - `"taiebat"`: K = K0 patm (1 + e)/e (p′/Pref)^exponent, patm = 0.101325 MPa (Sand default for unloading)
  - `"no_void_ratio"`: K = D (p′/Pref)^E
- New: `Get_virgin_loading_bulk_modulus(e, pe, model=None, params=None, Pref=101.325, calculate_ci=False, de=0, dpe=0)`,
  `Get_unloading_bulk_modulus(...)` (same arguments), `Set_bulk_modulus_law(phase, model, params, source)`.
  K is returned in MPa whatever the unit of `pe` and `Pref`.
- Removed: `Get_hardin_drnevich_K`, `Get_fitted_K`, `Update_hardin_K_params`, `NE34.Reset_Hardin_Drnevich_params`,
  attributes `KA`…`KE`, `dKA`…`dKE`.

| Old call | New call |
|---|---|
| `soil.Update_hardin_K_params("UL"); soil.Get_hardin_drnevich_K(pe)` | `soil.Get_virgin_loading_bulk_modulus(soil.e, pe)` |
| `soil.Update_hardin_K_params("Un"); soil.Get_hardin_drnevich_K(pe)` | `soil.Get_unloading_bulk_modulus(soil.e, pe)` (Taiebat form; old Hardin unloading set dropped) |
| `soil.Get_fitted_K(pe)` | `Get_virgin_loading_bulk_modulus` / `Get_unloading_bulk_modulus` with a `"no_void_ratio"` law |
| `soil.KA = x` | `soil.Get_..._bulk_modulus(e, pe, params={"A": (x, 0)})` or `soil.Set_bulk_modulus_law(...)` |
| `..., calculate_ci=True, dpe=d)` -> `[value, mini, maxi]` | `..., calculate_ci=True, dpe=d)` -> `(value, mini, maxi)` |

### v0.4.0

- Added soil science modules, copied verbatim from `11_Python_scripts`:
  `gfp_prog_tools.soils` (soil classes), `gfp_prog_tools.fluid_comp_models`,
  `gfp_prog_tools.gas_properties`, `gfp_prog_tools.water_properties`.
  Import them as submodules, e.g. `from gfp_prog_tools.soils import Karlsruhe_fine_sand`.

### v0.3.0

- New: `Round_value_uncertainty(value, unc, step=1, fallback_sig=3)` and
  `Format_value_uncertainty(value, unc, step=1, sep="±", fallback_sig=3)`: round a value and its
  uncertainty for display to a multiple of `step × 10^floor(log10|unc|)` (e.g. 62.2 ± 34.5 ->
  `60±30` with `step=1`, `60±35` with `step=0.5`).
- `__version__` now matches the package version.

### v0.2.2

- Confidence-interval functions now call `scipy.stats.t.interval` / `scipy.stats.norm.interval`
  directly (same results as v0.2.0).
- `claude/CLAUDE.md`: new "Trusted libraries before custom code" section.
- Install paths written as `%USERPROFILE%\gfp_prog_tools`, so the same files work on every computer.

### v0.2.0

All functions merged into a single module, `core.py`, with full docstrings.

Breaking changes:

- **Confidence intervals corrected.** Signatures are now `(mean, std, n, ci=0.95)` and
  `(popt, pcov, n, ci=0.95)`, with `ci` last and defaulting to 0.95.
  - Mean: the standard error is `std/sqrt(n)` and Student t uses `n - 1` degrees of freedom
    (v0.1.0 used the same `df` for both).
  - `curve_fit`: `sqrt(diag(pcov))` is already the standard error of each parameter, so it is
    no longer divided by `sqrt(df)`, which made v0.1.0 intervals too narrow by a factor √df.
    Student t now uses `n - p` degrees of freedom (`p` = number of fitted parameters).
    The normal version no longer needs `n`.
- `Set_graph_params(font="Times New Roman", style="whitegrid")`: the font is now applied
  (v0.1.0 fixed it to DejaVu Sans, and `sns.set_theme` then overrode it). Also sets the
  colorblind palette, uses the same font for math text, and embeds TrueType fonts in PDFs.
- `linestyles` is a list (same indexes as before); entries 11-14 were duplicates of 8 and are now distinct.
- `_wrap_marker`, `_wrap_linestyle`, `_limit_datetime_ticks` renamed to `Wrap_marker`,
  `Wrap_linestyle`, `Limit_datetime_ticks`.

Other changes:

- New: `Calculate_confidence_interval_from_sample`, `Save_figure`.
- `In_with_tol` now handles negative reference values.
- `Ensure_dir` uses `os.path.join` (portable) instead of a hard-coded backslash.

### v0.1.0

Verbatim copy of `General_functions.py` and `graph_parameters.py`.
