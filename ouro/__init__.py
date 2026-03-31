# OUROBOROS — Self-consuming feedback loop AI system
# Copyright (c) 2026 Sai Harsha Vinaykudikala. All rights reserved.
# PATENT PENDING

from .pipeline import OuroPipeline
from .config import OuroConfig
from .feedback import OuroFeedbackModule
from .infer import OuroInferenceEngine

__all__ = ["OuroPipeline", "OuroConfig", "OuroFeedbackModule", "OuroInferenceEngine"]
__version__ = "0.1.0"
