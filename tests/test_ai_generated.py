import pytest
import numpy as np
import hipmath

# AI-generated test: test_zero_length
def test_zero_length():
    assert hipmath.dot(np.array([]), np.array([])) == 0.0

