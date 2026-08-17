"""Native desktop GUI for Fast and Loop ML-AMPSIT studies."""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

from ampsit.analysis import run_timeseries_analysis, save_analysis_tables
from ampsit.config import load_config
from ampsit.emulation import generate_emulated_ensemble, validate_emulation_options
from ampsit.importance import importance_choices
from ampsit.plotting import DEFAULT_PLOT_KINDS, generate_analysis_figures
from ampsit.regressors import model_choices, resolve_model_key
from ampsit.runner import run_analysis_grid
from ampsit.transforms import transform_choices


PLOT_LABELS = {
    "performance": "Performance through time",
    "prediction": "Parity and residuals",
    "importance": "Method-specific importance",
    "temporal": "Temporal observed/predicted profile",
    "uncertainty": "Predictive uncertainty / disagreement",
    "manifold": "Manifold embedding",
    "ensemble": "Stacking consensus",
    "symbolic": "Symbolic equation / syntax tree / Pareto front",
}


def parse_int_selection(text: str) -> list[int]:
    """Parse ``1,3-5`` style selections while preserving input order."""
    values = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token[1:]:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    return list(dict.fromkeys(values))


def _default_path(filename):
    candidates = (Path.cwd() / filename, Path(__file__).resolve().parents[1] / filename)
    return next((path for path in candidates if path.exists()), candidates[0])


def merge_component_options(config, text):
    """Apply GUI-edited plugin options without mutating the loaded defaults."""
    parsed = json.loads(text or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Advanced component options must be a JSON object.")
    allowed = {
        "model_options", "tuning_spaces", "importance_options", "transform_options",
    }
    unknown = set(parsed) - allowed
    if unknown:
        raise ValueError(f"Unknown advanced option sections: {', '.join(sorted(unknown))}")
    for section, values in parsed.items():
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a JSON object.")
        config[section] = values
    return config


@dataclass
class FastStudy:
    model: str
    sample_count: int
    variable_index: int
    region_index: int
    vertical_level: int
    timestep: int
    tuning: int
    importance: str
    transform: str
    sobol_samples: int
    workers: int
    timesteps: tuple[int, ...]
    plot_kinds: tuple[str, ...]

    def validate(self, config):
        resolve_model_key(self.model)
        if not 8 <= self.sample_count <= int(config["totalsim"]):
            raise ValueError(f"Runs must be between 8 and {config['totalsim']}.")
        if not 1 <= self.variable_index <= len(config["variables"]):
            raise ValueError("Invalid variable selection.")
        if not 1 <= self.region_index <= len(config["regions"]):
            raise ValueError("Invalid region selection.")
        if not 1 <= self.vertical_level <= int(config["verticalmax"]):
            raise ValueError("Invalid vertical level.")
        if not 1 <= self.timestep <= int(config["totaltimesteps"]):
            raise ValueError("Invalid selected timestep.")
        if not self.timesteps or min(self.timesteps) < 1 or max(self.timesteps) > int(config["totaltimesteps"]):
            raise ValueError(f"Analyzed timesteps must be between 1 and {config['totaltimesteps']}.")
        if self.timestep not in self.timesteps:
            raise ValueError("The displayed timestep must be included in analyzed timesteps.")
        if self.workers < 1:
            raise ValueError("Workers must be at least one.")
        if not self.plot_kinds:
            raise ValueError("Select at least one plot family.")


class _ScrollableFrame:
    def __init__(self, parent, ttk, tk):
        self.canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.frame = ttk.Frame(self.canvas, padding=12)
        window = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.frame.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width))
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class AMPSITDesktopApp:
    def __init__(self, root, *, mode=None, config_path=None):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.root = root
        self.root.title("ML-AMPSIT Scientific Workbench")
        self.root.geometry("1040x820")
        self.root.minsize(900, 650)
        self.cancel_event = threading.Event()
        self.worker = None
        self.closing = False
        self.config_path = Path(config_path or _default_path("configAMPSIT.json"))
        self.config = load_config(self.config_path, resolve_paths=True)
        self.loop_config = dict(self.config.get("loop_study", {}))
        if mode in (None, "auto"):
            mode = self.config.get("run_mode", "fast")
        self._configure_style()
        self._build_header()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.fast_tab = ttk.Frame(self.notebook)
        self.loop_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.fast_tab, text="Fast study")
        self.notebook.add(self.loop_tab, text="Loop study")
        self._build_fast_tab()
        self._build_loop_tab()
        self._build_footer()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if mode == "loop":
            self.notebook.select(self.loop_tab)

    def _configure_style(self):
        style = self.ttk.Style(self.root)
        for theme in ("vista", "clam"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))

    def _build_header(self):
        frame = self.ttk.Frame(self.root, padding=(16, 12))
        frame.pack(fill="x")
        self.ttk.Label(frame, text="ML-AMPSIT", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Label(frame, text="Surrogate modelling, sensitivity and scientific diagnostics").grid(row=1, column=0, sticky="w")
        self.ttk.Button(frame, text="Load main JSON…", command=self._choose_main_config).grid(row=0, column=1, rowspan=2, padx=6)
        frame.columnconfigure(0, weight=1)

    def _section(self, parent, title, row, column=0, columnspan=1):
        section = self.ttk.LabelFrame(parent, text=title, padding=10, style="Section.TLabelframe")
        section.grid(row=row, column=column, columnspan=columnspan, sticky="nsew", padx=6, pady=6)
        return section

    def _combo(self, parent, label, values, row, default=0, *, state="readonly"):
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        variable = self.tk.StringVar(value=values[default] if values else "")
        widget = self.ttk.Combobox(parent, textvariable=variable, values=values, state=state, width=36)
        widget.grid(row=row, column=1, sticky="ew", pady=3)
        parent.columnconfigure(1, weight=1)
        return variable

    def _entry(self, parent, label, row, value, width=16):
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        variable = self.tk.StringVar(value=str(value))
        self.ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew", pady=3)
        parent.columnconfigure(1, weight=1)
        return variable

    def _plot_selector(self, parent, defaults=None, include_loop=False):
        defaults = set(defaults or DEFAULT_PLOT_KINDS)
        variables = {}
        for index, (key, label) in enumerate(PLOT_LABELS.items()):
            variable = self.tk.BooleanVar(value=key in defaults)
            self.ttk.Checkbutton(parent, text=label, variable=variable).grid(row=index // 2, column=index % 2, sticky="w", padx=4, pady=2)
            variables[key] = variable
        if include_loop:
            loop_choices = (
                ("spatial", "Loop vertical comparisons"),
                ("temporal_loop", "Loop temporal comparisons"),
                ("convergence", "Convergence with ensemble size N"),
            )
            for offset, (key, label) in enumerate(loop_choices, start=len(PLOT_LABELS)):
                variable = self.tk.BooleanVar(value=key in defaults)
                self.ttk.Checkbutton(parent, text=label, variable=variable).grid(row=offset // 2, column=offset % 2, sticky="w", padx=4, pady=2)
                variables[key] = variable
        return variables

    def _options_editor(self, parent):
        editor = self.tk.Text(parent, height=11, width=90, wrap="none", font=("Consolas", 9))
        editor.insert("1.0", json.dumps({
            "model_options": self.config.get("model_options", {}),
            "tuning_spaces": self.config.get("tuning_spaces", {}),
            "importance_options": self.config.get("importance_options", {}),
            "transform_options": self.config.get("transform_options", {}),
        }, indent=2))
        editor.pack(fill="both", expand=True)
        return editor

    def _build_fast_tab(self):
        scroll = _ScrollableFrame(self.fast_tab, self.ttk, self.tk)
        body = scroll.frame
        for column in (0, 1):
            body.columnconfigure(column, weight=1)
        analysis = self._section(body, "Analysis selection", 0, 0)
        model_labels = [label for label, _key in model_choices()]
        self._model_label_to_key = dict(model_choices())
        preset = dict(self.config.get("fast_study", {}))
        default_model_key = resolve_model_key(preset.get("model", "randomforest"))
        default_model_label = next((label for label, key in model_choices() if key == default_model_key), model_labels[0])
        self.fast_model = self._combo(analysis, "Regression model", model_labels, 0, model_labels.index(default_model_label))
        importance_labels = [label for label, _key in importance_choices()]
        self._importance_label_to_key = dict(importance_choices())
        default_importance = preset.get("importance_method", self.config.get("importance_method", "auto"))
        default_importance_label = next((label for label, key in importance_choices() if key == default_importance), importance_labels[0])
        self.fast_importance = self._combo(analysis, "Importance method", importance_labels, 1, importance_labels.index(default_importance_label))
        transform_labels = [label for label, _key in transform_choices()]
        self._transform_label_to_key = dict(transform_choices())
        default_transform = preset.get("feature_transform", self.config.get("feature_transform", "none"))
        default_transform_label = next((label for label, key in transform_choices() if key == default_transform), transform_labels[0])
        self.fast_transform = self._combo(analysis, "Feature representation", transform_labels, 2, transform_labels.index(default_transform_label))
        self.fast_variable = self._combo(analysis, "Variable", self.config["variables"], 3, int(preset.get("variable_index", 1)) - 1)
        self.fast_region = self._combo(analysis, "Region", self.config["regions"], 4, int(preset.get("region_index", 1)) - 1)

        sampling = self._section(body, "Sampling and execution", 0, 1)
        self.fast_runs = self._entry(sampling, "Simulations", 0, preset.get("sample_count", self.config["totalsim"]))
        self.fast_level = self._entry(sampling, "Vertical level", 1, preset.get("vertical_level", 1))
        self.fast_timestep = self._entry(sampling, "Displayed timestep", 2, preset.get("selected_timestep", self.config["totaltimesteps"]))
        self.fast_tune = self._combo(sampling, "Tuning", ["0 - configured", "1 - optimize", "2 - load tuned"], 3, int(preset.get("tuning", 0)))
        self.fast_sobol = self._entry(sampling, "Sobol base samples", 4, preset.get("sobol_samples", self.config.get("sobol_samples", 1024)))
        self.fast_workers = self._entry(sampling, "Parallel timesteps", 5, preset.get("parallel_workers", self.config.get("parallel_workers", 1)))
        preset_timesteps = preset.get("timesteps", [])
        self.fast_timesteps = self._entry(sampling, "Analyzed timesteps (blank = all)", 6, ",".join(map(str, preset_timesteps)))

        output = self._section(body, "Data and artefacts", 1, 0, 2)
        self.fast_data = self._entry(output, "Case data directory", 0, self.config["data_pathname"], 60)
        self.ttk.Button(output, text="Browse…", command=lambda: self._browse_directory(self.fast_data)).grid(row=0, column=2, padx=5)
        self.fast_output = self._entry(output, "Results directory", 1, self.config["output_pathname"], 60)
        self.ttk.Button(output, text="Browse…", command=lambda: self._browse_directory(self.fast_output)).grid(row=1, column=2, padx=5)
        self.fast_formats = self._entry(output, "Figure formats", 2, ",".join(self.config.get("plot_options", {}).get("formats", ["png"])))
        self.fast_dpi = self._entry(output, "Figure DPI", 3, self.config.get("plot_options", {}).get("dpi", 300))
        self.ttk.Label(output, text="Figures, tables, models and tuning reports are placed in analysis_outputs/<run>.", foreground="#455a64").grid(row=4, column=0, columnspan=3, sticky="w")

        plots = self._section(body, "Scientific figures", 2, 0, 2)
        self.fast_plots = self._plot_selector(plots, preset.get("plot_kinds", self.config.get("plot_options", {}).get("enabled")))
        emulation = self._section(body, "Emulated ensemble", 3, 0, 2)
        emulated = dict(self.config.get("emulated_ensemble", {}))
        self.fast_emulation_enabled = self.tk.BooleanVar(
            value=bool(emulated.get("enabled", False))
        )
        self.ttk.Checkbutton(
            emulation, text="Generate an emulated ensemble after validation",
            variable=self.fast_emulation_enabled,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=3)
        source_values = ["Sobol sequence", "User matrix"]
        source_default = 1 if str(emulated.get("source", "sobol")).lower() == "matrix" else 0
        self.fast_emulation_source = self._combo(
            emulation, "Input source", source_values, 1, source_default
        )
        self.fast_emulation_samples = self._entry(
            emulation, "Emulated samples", 2, emulated.get("sample_count", 256)
        )
        self.fast_emulation_input = self._entry(
            emulation, "User input matrix", 3, emulated.get("input_path", ""), 60
        )
        self.ttk.Button(
            emulation, text="Browse…",
            command=lambda: self._browse_file(self.fast_emulation_input),
        ).grid(row=3, column=2, padx=5)
        self.fast_emulation_levels = self._entry(
            emulation, "Levels (blank = selected)", 4,
            ",".join(map(str, emulated.get("levels", []))),
        )
        self.fast_emulation_timesteps = self._entry(
            emulation, "Timesteps (blank = analyzed)", 5,
            ",".join(map(str, emulated.get("timesteps", []))),
        )
        self.fast_emulation_plot_level = self._entry(
            emulation, "Spatial plot level", 6,
            emulated.get("plot_level", preset.get("vertical_level", 1)),
        )
        self.fast_emulation_plot_timestep = self._entry(
            emulation, "Spatial plot timestep", 7,
            emulated.get("plot_timestep", preset.get("selected_timestep", self.config["totaltimesteps"])),
        )
        advanced = self._section(body, "Advanced component options (JSON)", 4, 0, 2)
        self.ttk.Label(advanced, text="Edit model, importance and feature-transform options for this run only.", foreground="#455a64").pack(anchor="w", pady=(0, 4))
        self.fast_advanced = self._options_editor(advanced)
        self.fast_run_button = self.ttk.Button(body, text="Run Fast study", style="Run.TButton", command=self._run_fast)
        self.fast_run_button.grid(row=5, column=0, columnspan=2, pady=12)

    def _multi_list(self, parent, title, options, selected_keys, row, column):
        frame = self.ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        listbox = self.tk.Listbox(frame, selectmode="extended", exportselection=False, height=min(8, max(3, len(options))))
        mapping = []
        for index, (label, key) in enumerate(options):
            listbox.insert("end", label)
            mapping.append(key)
            if key in selected_keys:
                listbox.selection_set(index)
        listbox.pack(fill="both", expand=True)
        return listbox, mapping

    def _build_loop_tab(self):
        scroll = _ScrollableFrame(self.loop_tab, self.ttk, self.tk)
        body = scroll.frame
        for column in (0, 1, 2):
            body.columnconfigure(column, weight=1)
        configured_models = {
            resolve_model_key(value)
            for value in self.loop_config.get("models", ["randomforest"])
        }
        self.loop_models, self.loop_model_keys = self._multi_list(body, "Regression models", model_choices(), configured_models, 0, 0)
        configured_importance = set(self.loop_config.get("importance_methods", ["auto"]))
        self.loop_importances, self.loop_importance_keys = self._multi_list(body, "Importance methods", importance_choices(), configured_importance, 0, 1)
        configured_transforms = set(self.loop_config.get("feature_transforms", ["none"]))
        self.loop_transforms, self.loop_transform_keys = self._multi_list(body, "Feature representations", transform_choices(), configured_transforms, 0, 2)

        ranges = self._section(body, "Study grid (defaults loaded from the current JSON)", 1, 0, 3)
        self.loop_samples = self._entry(ranges, "Simulation counts", 0, ",".join(map(str, self.loop_config.get("sample_counts", []))))
        self.loop_levels = self._entry(ranges, "Vertical levels", 1, ",".join(map(str, self.loop_config.get("vertical_levels", []))))
        self.loop_variables, self.loop_variable_keys = self._multi_list(ranges, "Variables", [(value, index + 1) for index, value in enumerate(self.config["variables"])], set(self.loop_config.get("variable_indices", [1])), 0, 2)
        self.loop_regions, self.loop_region_keys = self._multi_list(ranges, "Regions", [(value, index + 1) for index, value in enumerate(self.config["regions"])], set(self.loop_config.get("region_indices", [1])), 1, 2)
        self.loop_timestep = self._entry(ranges, "Reference timestep for vertical profiles", 2, self.loop_config.get("selected_timestep", self.config["totaltimesteps"]))
        self.loop_tune = self._combo(ranges, "Tuning", ["0 - configured", "1 - optimize", "2 - load tuned"], 3, int(self.loop_config.get("tuning", 0)))
        self.loop_sobol = self._entry(ranges, "Sobol base samples", 4, self.loop_config.get("sobol_samples", self.config.get("sobol_samples", 1024)))
        self.loop_workers = self._entry(ranges, "Parallel configurations", 5, self.loop_config.get("parallel_workers", 2))
        self.loop_timesteps = self._entry(ranges, "Analyzed timesteps (blank = all)", 6, ",".join(map(str, self.loop_config.get("timesteps", []))))
        self.loop_data = self._entry(ranges, "Case data directory", 7, self.config["data_pathname"], 60)
        self.ttk.Button(ranges, text="Browse…", command=lambda: self._browse_directory(self.loop_data)).grid(row=7, column=2, padx=5, sticky="e")
        self.loop_output = self._entry(ranges, "Results directory", 8, self.config["output_pathname"], 60)
        self.ttk.Button(ranges, text="Browse…", command=lambda: self._browse_directory(self.loop_output)).grid(row=8, column=2, padx=5, sticky="e")
        self.loop_formats = self._entry(ranges, "Figure formats", 9, ",".join(self.config.get("plot_options", {}).get("formats", ["png"])))
        self.loop_dpi = self._entry(ranges, "Figure DPI", 10, self.config.get("plot_options", {}).get("dpi", 300))

        plots = self._section(body, "Figures for every run and Loop comparisons", 2, 0, 3)
        loop_plot_defaults = list(self.loop_config.get(
            "plot_kinds", self.config.get("plot_options", {}).get("enabled", ())
        )) + list(self.loop_config.get("comparison_kinds", ("spatial", "temporal", "convergence")))
        # The comparison uses a distinct UI key from the per-run temporal plot.
        loop_plot_defaults = ["temporal_loop" if value == "temporal" and value in self.loop_config.get("comparison_kinds", ()) else value for value in loop_plot_defaults]
        # Preserve the per-run temporal selection independently.
        if "temporal" in self.loop_config.get("plot_kinds", ()):
            loop_plot_defaults.append("temporal")
        self.loop_plots = self._plot_selector(
            plots,
            loop_plot_defaults,
            include_loop=True,
        )
        advanced = self._section(body, "Advanced component options (JSON)", 3, 0, 3)
        self.ttk.Label(advanced, text="These options are applied to every grid cell without modifying the default files.", foreground="#455a64").pack(anchor="w", pady=(0, 4))
        self.loop_advanced = self._options_editor(advanced)
        self.loop_run_button = self.ttk.Button(body, text="Run Loop study", style="Run.TButton", command=self._run_loop)
        self.loop_run_button.grid(row=4, column=0, columnspan=3, pady=12)

    def _build_footer(self):
        frame = self.ttk.Frame(self.root, padding=(12, 6, 12, 12))
        frame.pack(fill="x")
        self.progress = self.ttk.Progressbar(frame, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.status = self.tk.StringVar(value="Ready")
        self.ttk.Label(frame, textvariable=self.status, width=44).pack(side="left", padx=5)
        self.stop_button = self.ttk.Button(frame, text="STOP", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        self.open_button = self.ttk.Button(frame, text="Open output", command=self._open_output)
        self.open_button.pack(side="left", padx=4)

    def _selected(self, listbox, mapping):
        return [mapping[index] for index in listbox.curselection()]

    def _selected_plots(self, variables, *, loop=False):
        excluded = {"spatial", "temporal_loop", "convergence"} if loop else set()
        return tuple(key for key, variable in variables.items() if variable.get() and key not in excluded)

    def _runtime_config(self, data_value, output_value, formats, dpi, advanced_editor):
        config = json.loads(json.dumps(self.config))
        config["data_pathname"] = str(Path(data_value).expanduser())
        config["output_pathname"] = str(Path(output_value).expanduser())
        config.setdefault("plot_options", {})
        selected_formats = [value.strip().lower().lstrip(".") for value in formats.split(",") if value.strip()]
        if not selected_formats or any(value not in {"png", "pdf", "svg"} for value in selected_formats):
            raise ValueError("Figure formats must be a comma-separated subset of png, pdf, svg.")
        config["plot_options"]["formats"] = selected_formats
        config["plot_options"]["dpi"] = int(dpi)
        if config["plot_options"]["dpi"] < 50:
            raise ValueError("Figure DPI must be at least 50.")
        return merge_component_options(config, advanced_editor.get("1.0", "end").strip())

    def _read_fast(self):
        study = FastStudy(
            model=self._model_label_to_key[self.fast_model.get()],
            sample_count=int(self.fast_runs.get()),
            variable_index=self.config["variables"].index(self.fast_variable.get()) + 1,
            region_index=self.config["regions"].index(self.fast_region.get()) + 1,
            vertical_level=int(self.fast_level.get()), timestep=int(self.fast_timestep.get()),
            tuning=int(self.fast_tune.get().split()[0]),
            importance=self._importance_label_to_key[self.fast_importance.get()],
            transform=self._transform_label_to_key[self.fast_transform.get()],
            sobol_samples=int(self.fast_sobol.get()), workers=int(self.fast_workers.get()),
            timesteps=tuple(parse_int_selection(self.fast_timesteps.get()) or range(1, int(self.config["totaltimesteps"]) + 1)),
            plot_kinds=self._selected_plots(self.fast_plots),
        )
        config = self._runtime_config(
            self.fast_data.get(), self.fast_output.get(), self.fast_formats.get(), self.fast_dpi.get(), self.fast_advanced
        )
        study.validate(config)
        config["plot_options"]["enabled"] = list(study.plot_kinds)
        emulated_levels = parse_int_selection(self.fast_emulation_levels.get()) or [study.vertical_level]
        emulated_timesteps = parse_int_selection(self.fast_emulation_timesteps.get()) or list(study.timesteps)
        config["emulated_ensemble"] = {
            **dict(config.get("emulated_ensemble", {})),
            "enabled": bool(self.fast_emulation_enabled.get()),
            "source": "matrix" if self.fast_emulation_source.get() == "User matrix" else "sobol",
            "sample_count": int(self.fast_emulation_samples.get()),
            "input_path": self.fast_emulation_input.get().strip(),
            "levels": emulated_levels,
            "timesteps": emulated_timesteps,
            "plot_level": int(self.fast_emulation_plot_level.get()),
            "plot_timestep": int(self.fast_emulation_plot_timestep.get()),
        }
        validate_emulation_options(config, analyzed_timesteps=study.timesteps)
        return study, config

    def _read_loop(self):
        config = self._runtime_config(
            self.loop_data.get(), self.loop_output.get(), self.loop_formats.get(), self.loop_dpi.get(), self.loop_advanced
        )
        models = self._selected(self.loop_models, self.loop_model_keys)
        importance = self._selected(self.loop_importances, self.loop_importance_keys)
        transforms = self._selected(self.loop_transforms, self.loop_transform_keys)
        variables = self._selected(self.loop_variables, self.loop_variable_keys)
        regions = self._selected(self.loop_regions, self.loop_region_keys)
        if not all((models, importance, transforms, variables, regions)):
            raise ValueError("Select at least one model, importance, representation, variable, and region.")
        samples, levels = parse_int_selection(self.loop_samples.get()), parse_int_selection(self.loop_levels.get())
        if not samples or not levels:
            raise ValueError("Simulation counts and levels cannot be empty.")
        if min(samples) < 8 or max(samples) > int(config["totalsim"]):
            raise ValueError(f"Simulation counts must be between 8 and {config['totalsim']}.")
        if min(levels) < 1 or max(levels) > int(config["verticalmax"]):
            raise ValueError(f"Vertical levels must be between 1 and {config['verticalmax']}.")
        loop = {
            "models": models, "sample_counts": samples,
            "variable_indices": variables, "vertical_levels": levels,
            "region_indices": regions, "importance_methods": importance,
            "feature_transforms": transforms,
            "tuning": int(self.loop_tune.get().split()[0]),
            "selected_timestep": int(self.loop_timestep.get()),
            "sobol_samples": int(self.loop_sobol.get()),
            "parallel_workers": int(self.loop_workers.get()),
        }
        selected_timesteps = parse_int_selection(self.loop_timesteps.get())
        if selected_timesteps:
            if min(selected_timesteps) < 1 or max(selected_timesteps) > int(config["totaltimesteps"]):
                raise ValueError(f"Analyzed timesteps must be between 1 and {config['totaltimesteps']}.")
            loop["timesteps"] = selected_timesteps
            if loop["selected_timestep"] not in selected_timesteps:
                raise ValueError("The reference timestep must be included in analyzed timesteps.")
        if loop["parallel_workers"] < 1:
            raise ValueError("Parallel configurations must be at least one.")
        plot_kinds = self._selected_plots(self.loop_plots, loop=True)
        if not plot_kinds:
            raise ValueError("Select at least one per-run plot family.")
        comparisons = []
        if self.loop_plots["spatial"].get(): comparisons.append("spatial")
        if self.loop_plots["temporal_loop"].get(): comparisons.append("temporal")
        if self.loop_plots["convergence"].get(): comparisons.append("convergence")
        config["plot_options"]["enabled"] = list(plot_kinds)
        return config, loop, plot_kinds, comparisons

    def _start(self, target, description):
        if self.worker and self.worker.is_alive():
            return
        self.cancel_event.clear()
        self.progress.configure(value=0, maximum=100)
        self.status.set(description)
        self.stop_button.configure(state="normal")
        self.fast_run_button.configure(state="disabled")
        self.loop_run_button.configure(state="disabled")

        def wrapped():
            try:
                message = target()
            except Exception as error:
                detail = "".join(traceback.format_exception_only(type(error), error)).strip()
                self.root.after(0, lambda: self._finish(f"Error: {detail}", error=True))
            else:
                self.root.after(0, lambda: self._finish(message))

        self.worker = threading.Thread(target=wrapped, daemon=True, name="ampsit-gui-worker")
        self.worker.start()

    def _run_fast(self):
        try:
            study, config = self._read_fast()
        except Exception as error:
            self.messagebox.showerror("Invalid Fast study", str(error)); return

        def work():
            result = run_timeseries_analysis(
                config, model=study.model, sample_count=study.sample_count,
                variable_index=study.variable_index, region_index=study.region_index,
                vertical_level=study.vertical_level, tuning=study.tuning,
                importance_method=study.importance, feature_transform=study.transform,
                sobol_samples=study.sobol_samples, parallel_workers=study.workers,
                timesteps=study.timesteps,
                seed=int(config.get("random_seed", 42)), cancel_event=self.cancel_event,
            )
            save_analysis_tables(result, config, study.sample_count)
            figures = generate_analysis_figures(result, config, study.timestep, kinds=study.plot_kinds)
            emulated = generate_emulated_ensemble(
                result, config, cancel_event=self.cancel_event
            )
            self.progress_update(100)
            count = sum(len(paths) for paths in figures.values())
            if emulated is None:
                return f"Completed: {count} figure files in {result.artifact_dir}"
            emulated_figures = sum(len(paths) for paths in emulated.figures.values())
            return (
                f"Completed: {count} analysis figures, {emulated.sample_count} "
                f"emulated samples and {emulated_figures} emulation figures in "
                f"{result.artifact_dir}"
            )
        self._start(work, "Running Fast study…")

    def _run_loop(self):
        try:
            config, loop, plot_kinds, comparison_kinds = self._read_loop()
        except Exception as error:
            self.messagebox.showerror("Invalid Loop study", str(error)); return

        def progress(done, total):
            self.progress_update(100 * done / total)

        def work():
            output = run_analysis_grid(
                max_workers=loop["parallel_workers"], config=config, loop_config=loop,
                plot_kinds=plot_kinds, comparison_kinds=comparison_kinds,
                cancel_event=self.cancel_event, progress_callback=progress,
            )
            if output["cancelled"]:
                return f"Stopped after {len(output['runs'])} completed configurations."
            recovered = len(output.get("recovered_failures", ()))
            recovery_note = f" ({recovered} recovered after serial retry)" if recovered else ""
            return (
                f"Completed {len(output['runs'])} configurations and "
                f"{len(output['comparisons'])} comparison plots{recovery_note}."
            )
        self._start(work, "Running Loop study…")

    def progress_update(self, value):
        self.root.after(0, lambda: self.progress.configure(value=value))

    def _finish(self, message, error=False):
        if self.closing:
            self.root.destroy()
            return
        self.status.set(message)
        self.stop_button.configure(state="disabled")
        self.fast_run_button.configure(state="normal")
        self.loop_run_button.configure(state="normal")
        if error:
            self.messagebox.showerror("ML-AMPSIT", message)
        elif not self.cancel_event.is_set():
            self.messagebox.showinfo("ML-AMPSIT", message)

    def _stop(self):
        self.cancel_event.set()
        self.status.set("Cancellation requested; active fits will finish safely…")
        self.stop_button.configure(state="disabled")

    def _on_close(self):
        if self.worker is not None and self.worker.is_alive():
            if not self.messagebox.askyesno(
                "Close ML-AMPSIT",
                "An analysis is active. Request cancellation and close after active fits finish safely?",
            ):
                return
            self.closing = True
            self.cancel_event.set()
            self.status.set("Closing after active fits finish safely…")
            self.stop_button.configure(state="disabled")
            return
        self.root.destroy()

    def _browse_directory(self, variable):
        selected = self.filedialog.askdirectory(initialdir=variable.get() or str(Path.cwd()))
        if selected: variable.set(selected)

    def _browse_file(self, variable):
        selected = self.filedialog.askopenfilename(
            initialdir=str(Path(variable.get()).parent) if variable.get() else str(Path.cwd()),
            filetypes=(("Data matrices", "*.txt *.csv"), ("All files", "*.*")),
        )
        if selected:
            variable.set(selected)

    def _choose_main_config(self):
        if self.worker is not None and self.worker.is_alive():
            self.messagebox.showwarning("Analysis active", "Stop the active analysis before loading another configuration.")
            return
        selected = self.filedialog.askopenfilename(filetypes=(("JSON", "*.json"), ("All files", "*.*")))
        if selected:
            self.config_path = Path(selected)
            self._reload_interface()

    def _reload_interface(self, mode=None):
        config_path = self.config_path
        for child in self.root.winfo_children():
            child.destroy()
        self.__init__(self.root, mode=mode, config_path=config_path)

    def _open_output(self):
        active = self.notebook.index(self.notebook.select())
        value = self.fast_output.get() if active == 0 else self.loop_output.get()
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt": os.startfile(path)  # noqa: S606 - explicit user action
        else: self.messagebox.showinfo("Output directory", str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ML-AMPSIT desktop scientific workbench")
    parser.add_argument("--mode", choices=("auto", "fast", "loop"), default="auto")
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args(argv)
    import tkinter as tk
    root = tk.Tk()
    AMPSITDesktopApp(root, mode=arguments.mode, config_path=arguments.config)
    root.mainloop()


def main_fast():
    import sys
    main(["--mode", "fast", *sys.argv[1:]])


def main_loop():
    import sys
    main(["--mode", "loop", *sys.argv[1:]])


if __name__ == "__main__":
    main()
