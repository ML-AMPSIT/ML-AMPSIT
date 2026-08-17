from pathlib import Path

import numpy as np

from ampsit.analysis import run_timeseries_analysis, save_analysis_tables
from ampsit.desktop import FastStudy, merge_component_options, parse_int_selection
from ampsit.plotting import (
    applicable_plot_kinds,
    generate_analysis_figures,
    generate_loop_comparison_figures,
    importance_temporal_figure,
    performance_figure,
    temporal_prediction_figure,
)
from ampsit.runner import run_analysis_grid
import ampsit.runner as runner_module


def _case(tmp_path, *, levels=1, timesteps=3):
    rng = np.random.default_rng(123)
    x = rng.uniform([0.8, 8.0], [1.2, 12.0], size=(36, 2))
    np.savetxt(tmp_path / "X.txt", x)
    for level in range(1, levels + 1):
        for timestep in range(1, timesteps + 1):
            y = level + timestep * x[:, 0] + 0.1 * x[:, 1]
            np.savetxt(tmp_path / f"Y_site_lev{level}_{timestep}.txt", y[None, :], delimiter=",")
    return {
        "parameter_names": ["a", "b"],
        "MATRIX": [[1.0, 20.0], [10.0, 20.0]],
        "variables": ["Y"], "regions": ["site"],
        "verticalmax": levels, "totaltimesteps": timesteps, "totalsim": 36,
        "data_pathname": str(tmp_path), "output_pathname": str(tmp_path),
        "tun_iter": 1, "random_seed": 42,
        "plot_options": {"formats": ["png"], "dpi": 80, "close_after_save": True},
        "model_options": {
            "stacking": {"base_models": ["elasticnet", "cart"], "cv": 2, "n_jobs": 1}
        },
        "importance_options": {"pfi": {"n_repeats": 2, "n_jobs": 1}},
        "transform_options": {"kernel_pca": {"n_components": 2, "kernel": "rbf"}},
    }


def test_desktop_selection_parser_and_validation(tmp_path):
    config = _case(tmp_path)
    assert parse_int_selection("1, 3-5, 5") == [1, 3, 4, 5]
    study = FastStudy("lasso", 36, 1, 1, 1, 2, 0, "native", "none", 32, 1, (1, 2, 3), ("prediction",))
    study.validate(config)
    edited = merge_component_options(config.copy(), '{"model_options": {"mlp": {"max_iter": 25}}, "tuning_spaces": {"mlp": {"max_iter": {"type": "integer", "low": 50, "high": 100}}}}')
    assert edited["model_options"]["mlp"]["max_iter"] == 25
    assert edited["tuning_spaces"]["mlp"]["max_iter"]["high"] == 100


def test_manifold_and_probabilistic_plots(tmp_path):
    config = _case(tmp_path)
    result = run_timeseries_analysis(
        config, model="br", sample_count=36, variable_index=1,
        region_index=1, vertical_level=1, importance_method="pfi",
        feature_transform="kernel_pca", sobol_samples=16,
    )
    assert {"uncertainty", "manifold"} <= applicable_plot_kinds(result)
    save_analysis_tables(result, config, 36)
    saved = generate_analysis_figures(
        result, config, 2,
        kinds=("performance", "prediction", "importance", "temporal", "uncertainty", "manifold"),
    )
    assert "manifold" in saved
    assert "uncertainty_calibration" in saved
    assert all(path.stat().st_size > 0 for paths in saved.values() for path in paths)
    run_dir = Path(result.artifact_dir)
    assert (run_dir / "figures").is_dir()
    assert (run_dir / "tables").is_dir()
    assert (run_dir / "study_manifest.json").is_file()
    assert list((run_dir / "tables").glob("predictions_*.csv"))


def test_temporal_axes_remain_unit_agnostic(tmp_path):
    config = _case(tmp_path)
    config.update({
        "time_values": [0.0, 0.001, 0.002],
        "time_coordinate_name": "Time",
        "time_coordinate_units": "milliseconds",
    })
    result = run_timeseries_analysis(
        config, model="lasso", sample_count=36, variable_index=1,
        region_index=1, vertical_level=1, importance_method="native",
        sobol_samples=16,
    )
    figures = (
        performance_figure(result, config),
        temporal_prediction_figure(result, config),
        importance_temporal_figure(result, config["parameter_names"], config),
    )
    assert figures[0].axes[1].get_xlabel() == "Timestep"
    assert figures[1].axes[0].get_xlabel() == "Timestep"
    assert figures[2].axes[0].get_xlabel() == "Timestep"
    np.testing.assert_array_equal(figures[1].axes[0].lines[0].get_xdata(), [1, 2, 3])


def test_loop_end_to_end_builds_spatial_and_temporal_comparisons(tmp_path):
    config = _case(tmp_path, levels=2, timesteps=2)
    loop = {
        "models": ["lasso"], "sample_counts": [24, 36], "variable_indices": [1],
        "vertical_levels": [1, 2], "region_indices": [1],
        "importance_methods": ["native"], "feature_transforms": ["none"],
        "tuning": 0, "selected_timestep": 2, "sobol_samples": 16,
        "parallel_workers": 1,
    }
    output = run_analysis_grid(
        max_workers=2, config=config, loop_config=loop,
        plot_kinds=("prediction", "temporal"),
        comparison_kinds=("spatial", "temporal", "convergence"),
    )
    assert len(output["runs"]) == 4
    assert any(name.startswith("spatial__") for name in output["comparisons"])
    assert any(name.startswith("temporal__") for name in output["comparisons"])
    assert any(name.startswith("convergence__") for name in output["comparisons"])
    assert all(Path(path).stat().st_size > 0 for paths in output["comparisons"].values() for path in paths)
    assert (tmp_path / "analysis_outputs" / "loop_comparisons" / "loop_study_manifest.json").is_file()


def test_loop_retries_transient_process_failure_serially(tmp_path, monkeypatch):
    class FakeFuture:
        def __init__(self, function, argument):
            try:
                self.value = function(argument)
                self.error = None
            except Exception as error:
                self.value = None
                self.error = error

        def result(self):
            if self.error is not None:
                raise self.error
            return self.value

        def cancel(self):
            return False

    class FakeExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def submit(self, function, argument):
            return FakeFuture(function, argument)

        def shutdown(self, **_kwargs):
            pass

    calls = 0

    def transient_job(_argument):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary native-library failure")
        return {"artifact_dir": str(tmp_path), "figures": {}, "profile_records": []}

    monkeypatch.setattr(runner_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(runner_module, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(runner_module, "_execute_analysis_job", transient_job)
    monkeypatch.setattr(runner_module, "generate_loop_comparison_figures", lambda *_args, **_kwargs: {})
    config = {
        "data_pathname": str(tmp_path), "output_pathname": str(tmp_path),
        "random_seed": 42,
    }
    loop = {
        "models": ["lasso"], "sample_counts": [12], "variable_indices": [1],
        "vertical_levels": [1], "region_indices": [1], "importance_methods": ["native"],
        "feature_transforms": ["none"], "retry_failed_serially": True,
    }
    output = run_analysis_grid(
        max_workers=2, config=config, loop_config=loop,
        plot_kinds=(), comparison_kinds=(),
    )
    assert calls == 2
    assert len(output["runs"]) == 1
    assert output["recovered_failures"] == [["lasso", 12, 1, 1, 1, "native", "none"]]
    report = Path(output["failure_report"])
    assert "temporary native-library failure" in report.read_text(encoding="utf-8")


def test_stacking_gets_consensus_specific_figures(tmp_path):
    config = _case(tmp_path, timesteps=2)
    result = run_timeseries_analysis(
        config, model="stacking", sample_count=36, variable_index=1,
        region_index=1, vertical_level=1, importance_method="pfi",
        sobol_samples=16,
    )
    assert "ensemble" in applicable_plot_kinds(result)
    saved = generate_analysis_figures(result, config, 1, kinds=("ensemble", "uncertainty"))
    assert {"ensemble_members", "ensemble_correlation", "uncertainty_interval"} <= saved.keys()


def test_kernel_shap_gets_sample_distribution_figure(tmp_path):
    config = _case(tmp_path, timesteps=1)
    config["importance_options"]["kernel_shap"] = {"background": 4, "evaluations": 4}
    result = run_timeseries_analysis(
        config, model="lasso", sample_count=36, variable_index=1,
        region_index=1, vertical_level=1, importance_method="kernel_shap",
        sobol_samples=16,
    )
    importance = result.time_steps[0].evaluation.importance
    assert importance.attributions.shape[1] == 2
    saved = generate_analysis_figures(result, config, 1, kinds=("importance",))
    assert "shap_distribution" in saved


def test_symbolic_regression_gets_equation_tree_fit_and_pareto(tmp_path):
    config = _case(tmp_path, timesteps=1)
    config["model_options"]["symbolic"] = {
        "population_size": 40, "generations": 4, "max_depth": 4,
        "parsimony_coefficient": 1e-3,
    }
    result = run_timeseries_analysis(
        config, model="symbolic", sample_count=36, variable_index=1,
        region_index=1, vertical_level=1, importance_method="pfi",
        sobol_samples=16,
    )
    save_analysis_tables(result, config, 36)
    assert "symbolic" in applicable_plot_kinds(result)
    saved = generate_analysis_figures(result, config, 1, kinds=("symbolic",))
    assert {"symbolic_syntax_tree", "symbolic_fit", "symbolic_pareto_front"} <= saved.keys()
    run_dir = Path(result.artifact_dir)
    assert list((run_dir / "tables").glob("symbolic_equations_*.csv"))
    assert list((run_dir / "tables").glob("symbolic_pareto_*.csv"))
