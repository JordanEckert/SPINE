"""Comparator baseline methods."""

from .glvq import GLVQ
from .gng import GNG
from .lvq3 import LVQ3
from .reference import FullSet, KMeansPerClass, RandomSubsample
from .rsp3 import RSP3
from .spotgreedy import SPOTGreedy

__all__ = ["FullSet", "RandomSubsample", "KMeansPerClass",
           "RSP3", "LVQ3", "SPOTGreedy", "GLVQ", "GNG"]
