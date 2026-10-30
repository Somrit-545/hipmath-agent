import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _hipmath import saxpy, dot, gemv
__all__ = ["saxpy", "dot", "gemv"]