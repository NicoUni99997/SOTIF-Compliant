from analysis.pipeline import AnalysisPipeline
from analysis.rq2_coverage_entropy import RQ3CoverageEntropyAnalyzer
from analysis.rq2_event_percentage import RQ3EventPercentagesAnalyzer
from analysis.rq4_driving_style_non_collision import RQ4DrivingStyleNonCollisionAnalyzer
from pathlib import Path

if __name__ == "__main__":
    CSV_DIR = "./datasets"
    OUT_DIR = "./analysis/results"    

    pipe = AnalysisPipeline(csv_root_dir=CSV_DIR)
    # results = pipe.run_all(output_base_dir=OUT_DIR)

    # print("Analisi completata.")
    # print("RQ1 report:", results.rq1.report_txt_path)
    # print("RQ1 summary:", results.rq1.summary_csv_path)
    # print("RQ1 HR boxplot:", results.rq1.hr_boxplot_path)
    # print("RQ1 R boxplot:", results.rq1.r_boxplot_path)
    # print("RQ1 heatmap:", results.rq1.hazard_heatmap_path)

    # rq3_part1_dir = "analysis_outputs/rq3_part1_scenarios"

    # Parte 1: UMAP + KMeans su RUNS
    # rq3 = pipe.run_rq3_part1(output_dir=rq3_part1_dir, level="scenarios")
    # print("K*:", rq3.k_star)
    # print("Silhouette plot:", rq3.silhouette_plot_path)
    # print("UMAP hull plot:", rq3.umap_tool_hulls_plot_path)

    # Parte 2: Coverage + Entropy su RUNS
    # an = RQ3CoverageEntropyAnalyzer(csv_root_dir=Path(rq3_part1_dir), level="scenarios", min_count_per_cluster=1)
    # res = an.run(output_dir=Path(rq3_part1_dir))

    # print("RQ2 - CSV metrics:", res.metrics_csv_path)
    # print("RQ2 - Coverage plot:", res.coverage_plot_path)
    # print("RQ2 - Entropy plot::", res.entropy_plot_path)

    # Parte 3: Hazard percentage
    leaderboard_files = [
        "./datasets/ScenarioFuzzLLM_metrics_sotif_hazard_leaderboard.csv",
        "./datasets/SimADFuzz_metrics_sotif_hazard_leaderboard.csv",
        "./datasets/TMFuzz_metrics_sotif_hazard_leaderboard.csv",
    ]

    # an = RQ3EventPercentagesAnalyzer(leaderboard_files=leaderboard_files)
    # res = an.run(output_dir="analysis_outputs/rq3_part3")
    # print("RQ2 - CSV Percentages:", res.output_csv_path)

    an = RQ4DrivingStyleNonCollisionAnalyzer(
        leaderboard_files=leaderboard_files,
        driving_hazards=["lane_invasion", "off_road", "red_light", "stop_sign"],
        p_threshold=0.0
    )

    # rq4 = pipe.run_rq4_driving_style(output_dir="analysis_outputs/rq4", leaderboard_files=leaderboard_files)
    # print(rq4.summary_csv_path)
    # print(rq4.per_hazard_csv_path)
    # print(rq4.radar_plot_path)

    eff = pipe.run_efficiency_time_to_hazard(
        output_dir="analysis_outputs/efficiency",
        logs_dir="./datasets/"  # cartella dove stanno i *_log_basic.json
    )

    print(eff.per_hazard_csv_path)
    print(eff.hit_rates_plot_path)
    print(eff.mean_ttf_plot_path)
    print(eff.boxplot_path)