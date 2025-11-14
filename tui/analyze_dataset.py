import os
import pandas as pd
from tui.prob_plotting import boxplot_single, plot_empirical_cdf, plot_polygon_from_data
from variation_series import StatisticsAnalyzer


def analyze_dataset(csv_path: str = "DATA.csv", plot_dir: str = "plots"):
    os.makedirs(plot_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    numeric_df = df.select_dtypes(include=["number"])
    results = {}

    for column in numeric_df.columns:
        clean_data = numeric_df[column].dropna()
        if clean_data.empty:
            results[column] = {"error": "No valid data."}
            continue

        try:
            data_list = clean_data.tolist()
            analyzer = StatisticsAnalyzer(data_list)
            stats = {
                "count": len(clean_data),
                "mean": analyzer.get_expected_value_estimate(),
                "median": analyzer.get_median(),
                "mode": analyzer.get_mode(),
                "std": analyzer.get_sample_standard_deviation(),
                "std_corrected": analyzer.get_sample_standard_deviation_corrected(),
                "range": analyzer.get_whole_range(),
                "min": analyzer.get_extremes()[0],
                "max": analyzer.get_extremes()[1],
            }
            results[column] = stats

            safe_name = "".join(c if c.isalnum() else "_" for c in column)
            base_path = os.path.join(plot_dir, safe_name)

            # plot_polygon_from_data(data_list, f"{base_path}_polygon.png")
            # plot_empirical_cdf(data_list, f"{base_path}_ecdf.png")
            # boxplot_single(data_list, f"{base_path}_boxplot.png")

        except Exception as e:
            results[column] = {"error": str(e)}

    return results
