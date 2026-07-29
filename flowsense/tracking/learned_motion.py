"""Portable learned correction for FlowSense's constant-velocity forecast."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MODEL_FORMAT = "flowsense-ridge-scale-corrector-v1"


@dataclass(frozen=True, slots=True)
class LearnedCorrection:
    """Scale values produced by one learned correction."""

    learned_scale: float
    applied_scale: float


@dataclass(frozen=True, slots=True)
class LearnedMotionCorrector:
    """Evaluate the exported Ridge model without requiring scikit-learn."""

    horizon_frames: int
    velocity_window: int
    correction_strength: float
    scale_min: float
    scale_max: float
    class_categories: tuple[str, ...]
    numeric_features: tuple[str, ...]
    numeric_mean: tuple[float, ...]
    numeric_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_commit: str

    @classmethod
    def from_json(cls, path: str | Path) -> "LearnedMotionCorrector":
        """Load and strictly validate a portable corrector model."""
        model_path = Path(path)
        with model_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"Unsupported learned model format in {model_path}")

        classes = tuple(str(value) for value in payload["class_categories"])
        feature_names = tuple(str(value) for value in payload["numeric_features"])
        means = tuple(float(value) for value in payload["numeric_mean"])
        scales = tuple(float(value) for value in payload["numeric_scale"])
        coefficients = tuple(float(value) for value in payload["ridge_coefficients"])
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("Learned model classes must be unique and non-empty")
        if len(feature_names) != len(means) or len(means) != len(scales):
            raise ValueError("Learned model numeric feature arrays do not match")
        if len(coefficients) != len(classes) + len(feature_names):
            raise ValueError("Learned model coefficient count is inconsistent")
        if any(not math.isfinite(value) for value in (*means, *scales, *coefficients)):
            raise ValueError("Learned model contains a non-finite value")
        if any(value <= 0 for value in scales):
            raise ValueError("Learned model numeric scales must be positive")

        result = cls(
            horizon_frames=int(payload["horizon_frames"]),
            velocity_window=int(payload["velocity_window"]),
            correction_strength=float(payload["correction_strength"]),
            scale_min=float(payload["scale_min"]),
            scale_max=float(payload["scale_max"]),
            class_categories=classes,
            numeric_features=feature_names,
            numeric_mean=means,
            numeric_scale=scales,
            coefficients=coefficients,
            intercept=float(payload["ridge_intercept"]),
            training_commit=str(payload["training_commit"]),
        )
        if result.horizon_frames < 1 or result.velocity_window < 1:
            raise ValueError(
                "Learned model horizon and velocity window must be positive"
            )
        if not 0.0 <= result.correction_strength <= 1.0:
            raise ValueError(
                "Learned model correction strength must be between 0 and 1"
            )
        if result.scale_min > result.scale_max:
            raise ValueError("Learned model scale range is invalid")
        return result

    def compatible_with(self, *, horizon_frames: int, velocity_window: int) -> bool:
        """Return whether runtime prediction settings match training."""
        return (
            self.horizon_frames == horizon_frames
            and self.velocity_window == velocity_window
        )

    def correct(
        self,
        *,
        class_name: str,
        confidence: float,
        center_x: float,
        center_y: float,
        box_width: float,
        box_height: float,
        frame_width: float,
        frame_height: float,
        baseline_dx: float,
        baseline_dy: float,
        mean_velocity_x: float,
        mean_velocity_y: float,
        velocities: Sequence[tuple[float, float]],
        velocity_sample_seconds: float,
        history_points: int,
    ) -> LearnedCorrection:
        """Predict a cautious scale adjustment for one baseline arrow."""
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if box_width <= 0 or box_height <= 0:
            raise ValueError("Bounding-box dimensions must be positive")
        if not velocities:
            return LearnedCorrection(learned_scale=1.0, applied_scale=1.0)

        diagonal = math.hypot(frame_width, frame_height)
        acceleration_x = 0.0
        acceleration_y = 0.0
        if len(velocities) >= 2 and velocity_sample_seconds > 0:
            acceleration_x = (
                velocities[-1][0] - velocities[-2][0]
            ) / velocity_sample_seconds
            acceleration_y = (
                velocities[-1][1] - velocities[-2][1]
            ) / velocity_sample_seconds

        speed = math.hypot(mean_velocity_x, mean_velocity_y)
        segment_speeds = [math.hypot(x, y) for x, y in velocities]
        average_segment_speed = sum(segment_speeds) / len(segment_speeds)
        consistency = (
            speed / average_segment_speed if average_segment_speed > 0 else 0.0
        )
        direction = math.atan2(mean_velocity_y, mean_velocity_x)
        std_x = _population_std([velocity[0] for velocity in velocities])
        std_y = _population_std([velocity[1] for velocity in velocities])
        numeric = {
            "confidence": confidence,
            "center_x_normalized": center_x / frame_width,
            "center_y_normalized": center_y / frame_height,
            "width_normalized": box_width / frame_width,
            "height_normalized": box_height / frame_height,
            "aspect_ratio": box_width / max(box_height, 1e-6),
            "baseline_dx_normalized": baseline_dx / diagonal,
            "baseline_dy_normalized": baseline_dy / diagonal,
            "baseline_length_normalized": math.hypot(baseline_dx, baseline_dy)
            / diagonal,
            "speed_normalized": speed / diagonal,
            "acceleration_x_normalized": acceleration_x / diagonal,
            "acceleration_y_normalized": acceleration_y / diagonal,
            "acceleration_magnitude_normalized": math.hypot(
                acceleration_x, acceleration_y
            )
            / diagonal,
            "velocity_std_x_normalized": std_x / diagonal,
            "velocity_std_y_normalized": std_y / diagonal,
            "direction_sin": math.sin(direction),
            "direction_cos": math.cos(direction),
            "motion_consistency": consistency,
            "track_history_points": min(history_points, 30),
        }
        missing = set(self.numeric_features) - set(numeric)
        if missing:
            raise ValueError(f"Runtime is missing learned features: {sorted(missing)}")

        prediction = self.intercept
        if class_name in self.class_categories:
            prediction += self.coefficients[self.class_categories.index(class_name)]
        offset = len(self.class_categories)
        for index, name in enumerate(self.numeric_features):
            standardized = (
                numeric[name] - self.numeric_mean[index]
            ) / self.numeric_scale[index]
            prediction += self.coefficients[offset + index] * standardized

        learned_scale = min(self.scale_max, max(self.scale_min, prediction))
        applied_scale = 1.0 + self.correction_strength * (learned_scale - 1.0)
        return LearnedCorrection(
            learned_scale=learned_scale,
            applied_scale=applied_scale,
        )


def default_model_path() -> Path:
    """Return the bundled portable model path."""
    return (
        Path(__file__).resolve().parents[1]
        / "models"
        / "learned_prediction_corrector.json"
    )


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
