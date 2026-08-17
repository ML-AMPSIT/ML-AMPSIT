import numpy as np

from ampsit.analysis import run_timeseries_analysis
from ampsit.plotting import importance_detail_figures, prediction_figure


def test_end_to_end_native_analysis(tmp_path):
    rng = np.random.default_rng(11)
    x = rng.uniform([0.8, 8.0], [1.2, 12.0], size=(40, 2))
    np.savetxt(tmp_path / "X.txt", x)
    for timestep in (1, 2):
        y = timestep * x[:, 0] + 0.1 * x[:, 1]
        np.savetxt(tmp_path / f"Y_site_lev1_{timestep}.txt", y[None, :], delimiter=",")
    config = {
        "parameter_names": ["a", "b"],
        "MATRIX": [[1.0, 20.0], [10.0, 20.0]],
        "variables": ["Y"],
        "regions": ["site"],
        "verticalmax": 1,
        "totaltimesteps": 2,
        "data_pathname": str(tmp_path),
        "output_pathname": str(tmp_path),
        "tun_iter": 2,
        "random_seed": 42,
    }
    result = run_timeseries_analysis(
        config,
        model="lasso",
        sample_count=40,
        variable_index=1,
        region_index=1,
        vertical_level=1,
        importance_method="native",
        sobol_samples=32,
        parallel_workers=2,
    )
    assert list(result.metrics_frame.columns) == ["r2", "spearman_rho", "spearman_pvalue", "mse", "mae"]
    assert len(result.time_steps) == 2
    assert np.allclose(result.importance_frame.sum(axis=1), 1.0)


def test_fast_can_select_non_contiguous_temporal_profile_timesteps(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.uniform([0.8, 8.0], [1.2, 12.0], size=(32, 2))
    np.savetxt(tmp_path / "X.txt", x)
    for timestep in (1, 2, 3):
        np.savetxt(
            tmp_path / f"Y_site_lev1_{timestep}.txt",
            (timestep * x[:, 0] + 0.1 * x[:, 1])[None, :], delimiter=",",
        )
    config = {
        "parameter_names": ["a", "b"], "MATRIX": [[1.0, 20.0], [10.0, 20.0]],
        "variables": ["Y"], "regions": ["site"], "verticalmax": 1,
        "totaltimesteps": 3, "data_pathname": str(tmp_path),
        "output_pathname": str(tmp_path), "tun_iter": 1,
    }
    result = run_timeseries_analysis(
        config, model="lasso", sample_count=32, variable_index=1,
        region_index=1, vertical_level=1, importance_method="native",
        sobol_samples=16, timesteps=[1, 3],
    )
    assert [item.timestep for item in result.time_steps] == [1, 3]
    figure = prediction_figure(result, 3, config)
    assert figure is not None


def test_tuning_artifact_names_do_not_repeat_run_metadata(tmp_path):
    rng = np.random.default_rng(23)
    x = rng.uniform([0.8, 8.0], [1.2, 12.0], size=(32, 2))
    np.savetxt(tmp_path / "X.txt", x)
    y = x[:, 0] + 0.1 * x[:, 1]
    np.savetxt(
        tmp_path / "response_benchmark_lev1_1.txt", y[None, :], delimiter=",",
    )
    config = {
        "parameter_names": ["a", "b"],
        "MATRIX": [[1.0, 20.0], [10.0, 20.0]],
        "variables": ["response"], "regions": ["benchmark"],
        "verticalmax": 1, "totaltimesteps": 1,
        "data_pathname": str(tmp_path),
        "output_pathname": str(tmp_path), "tun_iter": 1,
        "tuning_spaces": {"cart": {}},
        "importance_options": {"pfi": {"n_repeats": 2, "n_jobs": 1}},
    }

    result = run_timeseries_analysis(
        config, model="cart", sample_count=32, variable_index=1,
        region_index=1, vertical_level=1, tuning=1,
        importance_method="pfi", feature_transform="diffusion_maps",
        sobol_samples=16,
    )

    run_dir = result.artifact_dir
    assert (run_dir / "models" / "model_timestep1.joblib").is_file()
    assert (run_dir / "reports" / "tuning_results_timestep1.txt").is_file()
    from ampsit.analysis import save_analysis_tables
    save_analysis_tables(result, config, 32)
    assert (run_dir / "tables" / "metrics_results.csv").is_file()
    assert (run_dir / "tables" / "predictions_test.csv").is_file()


def test_bayesian_sobol_plot_layout(tmp_path):
    rng = np.random.default_rng(12)
    x = rng.uniform([0.8, 8.0], [1.2, 12.0], size=(32, 2))
    np.savetxt(tmp_path / "X.txt", x)
    for timestep in (1, 2):
        y = timestep * x[:, 0] + 0.1 * x[:, 1]
        np.savetxt(tmp_path / f"Y_site_lev1_{timestep}.txt", y[None, :], delimiter=",")
    config = {
        "parameter_names": ["a", "b"],
        "MATRIX": [[1.0, 20.0], [10.0, 20.0]],
        "variables": ["Y"],
        "regions": ["site"],
        "verticalmax": 1,
        "totaltimesteps": 2,
        "data_pathname": str(tmp_path),
        "output_pathname": str(tmp_path),
        "tun_iter": 1,
        "random_seed": 42,
    }
    result = run_timeseries_analysis(
        config,
        model="br",
        sample_count=32,
        variable_index=1,
        region_index=1,
        vertical_level=1,
        importance_method="sobol",
        sobol_samples=16,
    )
    figures = importance_detail_figures(result, config, 1)
    assert {"sobol_orders", "sobol_interactions"} <= set(figures)
    figures["sobol_interactions"].savefig(tmp_path / "bayesian_sobol.png")
    assert (tmp_path / "bayesian_sobol.png").stat().st_size > 0
