"""Object tracking utilities."""

from .bytetrack_tracker import ByteTrackTracker
from .identity_consolidator import ConsolidationResult, IdentityConsolidator
from .learned_motion import LearnedCorrection, LearnedMotionCorrector, default_model_path
from .motion_prediction import MotionPoint, MotionPredictor, MotionSnapshot
from .schemas import Detection, TrackedDetection

__all__ = [
    "ByteTrackTracker",
    "ConsolidationResult",
    "Detection",
    "IdentityConsolidator",
    "LearnedCorrection",
    "LearnedMotionCorrector",
    "MotionPoint",
    "MotionPredictor",
    "MotionSnapshot",
    "TrackedDetection",
    "default_model_path",
]
