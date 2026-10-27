# gfp_prog_tools

General Python helper functions: data handling, confidence intervals, graph formatting.

```python
from gfp_prog_tools import Set_graph_params, Save_figure, Wrap_marker, Calculate_student_t_confidence_interval_curve_fit
```

## Install

On the computer where the package is developed (edits take effect immediately):

```
pip install -e C:\Users\floodpage\gfp_prog_tools
```

Anywhere else, or as a dependency in a project's `requirements.txt`, pin a version tag:

```
gfp_prog_tools @ git+https://github.com/gfp96/gfp_prog_tools.git@v0.2.0
```

## Claude Code conventions

[claude/CLAUDE.md](claude/CLAUDE.md) holds the Python conventions Claude Code should follow.
Copy it to `%USERPROFILE%\.claude\CLAUDE.md` on each computer so it applies in every project
(append it if that file already exists).

## Versions

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
