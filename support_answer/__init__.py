"""Grounded support-answer service: retrieve, check support, answer or abstain."""

from .config import Settings
from .pipeline import SupportAnswerService

__all__ = ["Settings", "SupportAnswerService"]
__version__ = "1.0.0"
