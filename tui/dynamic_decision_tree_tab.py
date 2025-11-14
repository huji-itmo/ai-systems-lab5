import numpy as np
import pandas as pd
import math
import random
from typing import Dict, List, Any, Optional, cast
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll, Grid
from textual.widgets import Static, Input, Label, Checkbox, Button
from textual.validation import Number
from textual.widget import Widget

from tui.helper import sanitize_id
from tui.decision_tree import DecisionTreeC45


class DynamicDecisionTreeTab(Static):
    DEFAULT_CSS = """
        .section {
            margin: 1 0;
            padding: 1;
            border: solid $primary;
            height: auto;
            min-height: 6;
        }

        #feature-checkboxes {
            grid-size: 4;
            grid-gutter: 1;
            height: auto;
        }

        #dynamic-inputs {
            grid-size: 2;
            grid-gutter: 1;
            height: auto;
        }

        #prediction-display {
            padding: 1;
            border: heavy $success;
            margin-top: 1;
            text-align: center;
        }

        Button {
            margin-top: 1;
        }
    """

    GRADE_MAPPING = {
        0: "Fail",
        1: "DD",
        2: "DC",
        3: "CC",
        4: "CB",
        5: "BB",
        6: "BA",
        7: "AA",
    }

    def __init__(
        self,
        model_name: str,
        all_feature_names: List[str],
        x_data: List[List[float]],
        y_data: List[float],
        initial_values: Optional[Dict[str, float]] = None,
        target_name: str = "Target",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.all_feature_names = all_feature_names
        self.x_data_full = np.array(x_data, dtype=float)
        self.y_data = np.array(y_data, dtype=float)
        self.initial_values = initial_values or {}
        self.target_name = target_name
        self._tree: Optional[DecisionTreeC45] = None
        self.default_k = 5  # Default k for prediction confidence

        # Select sqrt(n) random features by default
        n_features = len(all_feature_names)
        default_count = max(1, int(math.sqrt(n_features)))
        self.default_selected = random.sample(
            all_feature_names, min(default_count, n_features)
        )
        self.selected_features = set(self.default_selected)

    @property
    def tree(self) -> Optional[DecisionTreeC45]:
        return self._tree

    @tree.setter
    def tree(self, value: Optional[DecisionTreeC45]) -> None:
        self._tree = value

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            # Section 1: Feature selection
            with Vertical(classes="section") as feature_section:
                feature_section.border_title = "✅ Select Features"
                with Grid(id="feature-checkboxes"):
                    for feat in self.all_feature_names:
                        is_default = feat in self.default_selected
                        cb = Checkbox(
                            feat, value=is_default, id=f"cb-{sanitize_id(feat)}"
                        )
                        yield cb

                yield Button("Build Decision Tree", variant="success", id="build-tree")

            # Section 2: Input values
            with Vertical(classes="section") as input_section:
                input_section.border_title = "🩺 Input Values"
                self.inputs_container = Grid(id="dynamic-inputs")
                yield self.inputs_container

            # Section 3: Prediction
            with Vertical(classes="section") as pred_section:
                pred_section.border_title = "🔍 Prediction"
                yield Label(
                    "Build a tree first to see predictions", id="prediction-display"
                )

    def on_mount(self) -> None:
        self._refresh_inputs()
        self.call_later(self.build_tree)

    @on(Checkbox.Changed)
    def on_checkbox_changed(self, event: Checkbox.Changed):
        feat = event.checkbox.label.plain
        if event.checkbox.value:
            self.selected_features.add(feat)
        else:
            self.selected_features.discard(feat)
        self._refresh_inputs()

    @on(Button.Pressed, "#build-tree")
    def build_tree(self):
        if not self.selected_features:
            prediction_display = self.query_one("#prediction-display", Label)
            prediction_display.update("⚠️ Please select at least one feature")
            return

        try:
            # Prepare data for selected features
            selected_list = sorted(
                self.selected_features, key=lambda x: self.all_feature_names.index(x)
            )
            col_indices = [self.all_feature_names.index(f) for f in selected_list]
            x_data_selected = self.x_data_full[:, col_indices]

            # Create feature names mapping
            feature_names = [self.all_feature_names[i] for i in col_indices]

            # Build the tree
            self.tree = DecisionTreeC45(max_depth=5, min_samples_split=5)
            self.tree.fit(x_data_selected, self.y_data, feature_names)

            # Update prediction display
            self.update_prediction_display()

        except Exception as e:
            prediction_display = self.query_one("#prediction-display", Label)
            prediction_display.update(f"❌ Error building tree: {str(e)}")

    def _refresh_inputs(self):
        self.inputs_container.remove_children()

        for feat in self.all_feature_names:
            if feat not in self.selected_features:
                continue

            label = Label(f"{feat}:")
            validator = Number(minimum=0)  # All features ≥ 0
            safe_id = f"input-{sanitize_id(feat)}"
            input_widget = Input(
                placeholder=f"Enter {feat}",
                validators=[validator],
                id=safe_id,
            )
            if feat in self.initial_values:
                if isinstance(input_widget, Input):
                    input_widget.value = str(self.initial_values[feat])
            self.inputs_container.mount(Horizontal(label, input_widget))

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed):
        if self.tree:
            self.update_prediction_display()

    def get_grade_label(self, grade_value: int) -> str:
        """Convert numeric grade to letter grade according to specification"""
        return self.GRADE_MAPPING.get(
            int(grade_value), f"Unknown Grade ({grade_value})"
        )

    def update_prediction_display(self):
        if not self.tree or not self.selected_features:
            return

        try:
            # Get input values for selected features
            selected_list = sorted(
                self.selected_features, key=lambda x: self.all_feature_names.index(x)
            )
            input_values = []
            for feat in selected_list:
                input_widget = self.query_one(f"#input-{sanitize_id(feat)}", Input)
                val = input_widget.value.strip()
                if not val:
                    raise ValueError(f"Missing value for {feat}")
                input_values.append(float(val))

            # Create sample array
            sample = np.array(input_values).reshape(1, -1)

            # Get prediction and confidence
            prediction, confidence = self.tree.predict_with_confidence(
                sample, k=self.default_k
            )

            if prediction is None:
                raise ValueError("Prediction failed")

            # Format prediction display using grade mapping
            grade_label = self.get_grade_label(int(prediction))
            confidence_percent = confidence * 100

            display_text = (
                f"Predicted {self.target_name}: [bold]{grade_label}[/bold]\n"
                f"Confidence: {confidence_percent:.1f}%"
            )

            # Add tree path explanation
            path_explanation = self.tree.get_prediction_path(sample[0])
            if path_explanation:
                display_text += f"\n\nPath:\n{path_explanation}"

            prediction_display = self.query_one("#prediction-display", Label)
            prediction_display.update(display_text)

        except Exception as e:
            prediction_display = self.query_one("#prediction-display", Label)
            prediction_display.update(f"⚠️ Prediction error: {str(e)}")
