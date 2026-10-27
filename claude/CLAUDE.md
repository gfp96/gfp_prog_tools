# Python programming conventions

Apply these rules whenever writing or modifying Python code.

## Helper package: gfp_prog_tools

- Source: `C:\Users\floodpage\gfp_prog_tools` (GitHub: https://github.com/gfp96/gfp_prog_tools).
- Import with `from gfp_prog_tools import ...`. If it is missing from the active environment,
  install it: `pip install -e C:\Users\floodpage\gfp_prog_tools` on this computer, otherwise
  `pip install "gfp_prog_tools @ git+https://github.com/gfp96/gfp_prog_tools.git@v0.2.0"`.
- Before writing a utility, check whether gfp_prog_tools already provides it
  (read `src/gfp_prog_tools/core.py`), and use it instead of re-implementing it. Available:
  - Files/data: `Ensure_dir`, `dateparse`, `Eliminate_absurd_values`, `Get_index_min_value`,
    `In_with_tol`, `Get_nb_files`, `Get_indexes_compare_vectors`, `Update_df`,
    `Resample_data_frame_scipy`, `Resample_data_frame_linear_regression`
  - Confidence intervals: `Calculate_student_t_confidence_interval(mean, std, n, ci)`,
    `Calculate_normal_confidence_interval(mean, std, n, ci)`,
    `Calculate_confidence_interval_from_sample(data, ci, distribution)`,
    `Calculate_student_t_confidence_interval_curve_fit(popt, pcov, n, ci)`,
    `Calculate_normal_confidence_interval_curve_fit(popt, pcov, ci)`
  - Graphs: `Set_graph_params(font, style)`, `Save_figure(fig, folder, name, fmt)`,
    `Limit_datetime_ticks`, `markers`, `linestyles`, `Wrap_marker(i)`, `Wrap_linestyle(i)`
- If a helper looks wrong or is missing something generally useful, say so instead of
  silently working around it.

## Comments and docstrings

- Inline comment on every line that introduces a new variable, briefly saying what it is
  or what the equation computes.
- Full-line comment above complex equations giving, where relevant: the equation's name,
  its bibliographic source, and a human-readable form. Example:
  ```python
  # Hardin & Drnevich (1972) hyperbolic model: G/Gmax = 1 / (1 + γ/γr)
  G_ratio = 1/(1+gamma/gamma_r)  # normalised shear modulus
  ```
- Every function has a docstring: general description, then a `Parameters` section
  (numpy style) explaining each parameter, then `Returns`.

## Graphs

- Use matplotlib (+ seaborn). Call `Set_graph_params(font=font)` before creating figures.
- Individual colours: `sns.color_palette("colorblind")`.
- Continuous colour maps: `"viridis"` for one-directional data; `"coolwarm"` for
  diverging data (values on either side of a meaningful centre, e.g. ± around zero).
- Markers and line styles: `Wrap_marker(i)` / `Wrap_linestyle(i)` for series i.
- Every function that creates a graph takes the parameters `fmt="pdf"` (saved file
  extension) and `font="Times New Roman"`, and saves with `Save_figure(fig, folder, name, fmt=fmt)`.

## Statistics

- Compute confidence intervals with the gfp_prog_tools functions above.
- Any function computing a confidence interval takes a parameter `ci=0.95`.
- For `curve_fit` results: Student t version with the default `absolute_sigma=False`;
  normal version with `absolute_sigma=True`.
