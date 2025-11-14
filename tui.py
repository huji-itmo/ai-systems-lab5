import numpy as np
import pandas as pd
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Grid
from textual.widgets import (
    TabbedContent,
    TabPane,
    Footer,
    Header,
)
from textual.binding import Binding

from tui.dynamic_decision_tree_tab import DynamicDecisionTreeTab
from tui.analyze_dataset import analyze_dataset
from tui.stat_panel import StatPanel
from tui.theme import get_theme


class MyApp(App):
    CSS = """

    TabbedContent {
        height: 100%;
    }
    .title {
        margin-top: 1;
        text-style: bold;
    }
    Horizontal {
        height: 3;
        align: left middle;
    }
    Input {
        width: 30;
        margin-left: 2;
    }
    #prediction {
        margin-top: 1;
        padding: 1 2;
        background: $surface;
        border: tall $primary;
    }
    .error {
        color: $error;
    }

    #stats_grid {
        grid-size: 3;
        grid-gutter: 1;
        height: auto;
    }
    """

    # Optional: Define key hints for the footer
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "unfocus", "Unfocus", show=False),  # <-- Add this
    ]

    def __init__(self, csv_path: str = "DATA.csv"):
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) > 1:
            self.all_features = numeric_columns[
                :-1
            ]  # All numeric columns except the last one
            self.target = numeric_columns[-1]  # Last numeric column as target
        else:
            # Fallback: if there are not enough numeric columns, use all except the last column
            all_cols = self.df.columns.tolist()
            self.all_features = all_cols[:-1]
            self.target = all_cols[-1]

        # Sample one random row for initial values
        random_row = self.df.sample(n=1).iloc[0]
        self.model1_initial = {feat: random_row[feat] for feat in self.all_features}

        # Model 1: Full
        self.X_all = self.df[self.all_features].to_numpy()
        self.y = self.df[self.target].to_numpy()

        # Analyze dataset
        self.analysis_results = analyze_dataset(csv_path)

    def on_mount(self) -> None:
        self.register_theme(get_theme())
        self.theme = "monokai"

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("📊 Dataset Statistics"):
                with VerticalScroll():
                    with Grid(id="stats_grid"):
                        for var_name, stats in self.analysis_results.items():
                            yield StatPanel(var_name=var_name, stats=stats)

            # Add the new Decision Tree tab
            with TabPane("🌳 Decision Tree (C4.5)"):
                yield DynamicDecisionTreeTab(
                    model_name="Decision Tree",
                    all_feature_names=self.all_features,
                    x_data=self.X_all.tolist(),
                    y_data=self.y.tolist(),
                    initial_values=self.model1_initial,
                    target_name=self.target,
                )

        yield Footer()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


if __name__ == "__main__":
    MyApp("DATA.csv").run()
