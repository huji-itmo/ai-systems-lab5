import numpy as np
import math
from collections import Counter
from typing import Optional, Dict, List, Tuple, Any


class Node:
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[Any] = None,
        feature_name: Optional[str] = None,
    ):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.feature_name = feature_name  # Name of the feature (for display)
        self.threshold = threshold  # Threshold value for split
        self.left = left  # Left subtree (<= threshold)
        self.right = right  # Right subtree (> threshold)
        self.value = value  # Class label if leaf node
        self.samples = 0  # Number of samples at this node
        self.class_counts: Dict[Any, int] = {}  # Class distribution at this node


class DecisionTreeC45:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 5,
        min_gain_ratio: float = 0.01,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_ratio = min_gain_ratio
        self.root: Optional[Node] = None
        self.feature_names: List[str] = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None
    ):
        """Build the decision tree"""
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names

        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the decision tree"""
        node = Node()
        node.samples = len(y)
        node.class_counts = dict(Counter(y))

        # Base cases
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node.value = self._most_common_class(y)
            return node

        # Find best split
        best_feature, best_threshold, best_gain_ratio = self._find_best_split(X, y)

        # If no good split found or gain ratio is too small
        if best_feature is None or best_gain_ratio < self.min_gain_ratio:
            node.value = self._most_common_class(y)
            return node

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Create child nodes
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Create current node
        node.feature_idx = best_feature
        node.feature_name = self.feature_names[best_feature]
        node.threshold = best_threshold
        node.left = left_node
        node.right = right_node

        return node

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best feature and threshold to split on"""
        best_gain_ratio = -1.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        # For each feature
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]

            # Get unique values and sort them
            unique_values = np.unique(feature_values)

            # If only one unique value, skip this feature
            if len(unique_values) <= 1:
                continue

            # Try each possible threshold between unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Skip if split is too uneven
                if (
                    np.sum(left_mask) < self.min_samples_split
                    or np.sum(right_mask) < self.min_samples_split
                ):
                    continue

                # Calculate gain ratio
                gain_ratio = self._calculate_gain_ratio(y, left_mask, right_mask)

                # Update best split
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain_ratio

    def _calculate_gain_ratio(
        self, y: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray
    ) -> float:
        """Calculate gain ratio for a split"""
        # Calculate information gain
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])

        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        information_gain = parent_entropy - weighted_entropy

        # Calculate split info
        split_info = self._split_info(n_left, n_right, n)

        # Avoid division by zero
        if split_info == 0 or information_gain <= 0:
            return 0.0

        return information_gain / split_info

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a set of labels"""
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def _split_info(self, n_left: int, n_right: int, n_total: int) -> float:
        """Calculate split info for gain ratio normalization"""
        if n_left == 0 or n_right == 0 or n_total == 0:
            return 0.0

        p_left = n_left / n_total
        p_right = n_right / n_total

        return -p_left * np.log2(p_left + 1e-9) - p_right * np.log2(p_right + 1e-9)

    def _most_common_class(self, y: np.ndarray) -> Any:
        """Return the most common class in a set of labels"""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for samples in X"""
        if self.root is None:
            raise ValueError("Tree has not been built. Call fit() first.")

        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample: np.ndarray) -> Any:
        """Predict class for a single sample"""
        if self.root is None:
            raise ValueError("Tree has not been built. Call fit() first.")

        node = self.root
        while node is not None and node.value is None and node.feature_idx is not None:
            if sample[node.feature_idx] <= node.threshold:
                if node.left is not None:
                    node = node.left
                else:
                    break
            else:
                if node.right is not None:
                    node = node.right
                else:
                    break
        return node.value if node is not None else None

    def predict_with_confidence(self, X: np.ndarray, k: int = 5) -> Tuple[Any, float]:
        """
        Predict class with confidence estimation
        Returns: (prediction, confidence)
        """
        if self.root is None:
            raise ValueError("Tree has not been built. Call fit() first.")

        node = self.root
        sample = X[0]  # Assume single sample

        # Traverse to leaf node
        while node is not None and node.value is None and node.feature_idx is not None:
            if sample[node.feature_idx] <= node.threshold:
                if node.left is not None:
                    node = node.left
                else:
                    break
            else:
                if node.right is not None:
                    node = node.right
                else:
                    break

        if node is None:
            return None, 0.0

        # Get prediction
        prediction = node.value

        # Calculate confidence based on class distribution in leaf
        total = node.samples
        correct = node.class_counts.get(prediction, 0)
        confidence = correct / total if total > 0 else 0.5

        return prediction, confidence

    def get_prediction_path(self, sample: np.ndarray) -> str:
        """Get human-readable path for a prediction"""
        if self.root is None:
            return "Tree has not been built. Call fit() first."

        node = self.root
        path_lines = []

        while node is not None and node.value is None and node.feature_idx is not None:
            feature_val = sample[node.feature_idx]
            condition = "<=" if feature_val <= node.threshold else ">"
            feature_name = node.feature_name or f"Feature_{node.feature_idx}"
            path_lines.append(f"{feature_name} {condition} {node.threshold:.2f}")

            if feature_val <= node.threshold:
                if node.left is not None:
                    node = node.left
                else:
                    break
            else:
                if node.right is not None:
                    node = node.right
                else:
                    break

        if node is None:
            return "Prediction path ended unexpectedly."

        # Add leaf information
        class_counts_str = ", ".join(
            [f"{cls}: {count}" for cls, count in node.class_counts.items()]
        )
        path_lines.append(
            f"Leaf node (samples: {node.samples}, classes: {class_counts_str})"
        )
        path_lines.append(f"Prediction: {node.value}")

        return "\n".join(path_lines)
