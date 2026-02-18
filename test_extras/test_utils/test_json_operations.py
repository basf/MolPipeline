"""Unit test for json operations with torch."""

import unittest

import numpy as np
import torch

from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json


class TestJsonOperations(unittest.TestCase):
    """Test serialization and deserialization of torch.Tensors to and from JSON."""

    def test_json_round_trip_tensor(self) -> None:
        """Test that round-trip results in the same Tensor."""
        original_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

        json_data = recursive_to_json(original_tensor)
        deserialized_tensor = recursive_from_json(json_data)

        self.assertTrue(torch.equal(original_tensor, deserialized_tensor))

    def test_json_round_trip_tensor_integer(self) -> None:
        """Test that round-trip results in the same Tensor with a single value."""
        original_tensor = torch.from_numpy(np.array(1))

        json_data = recursive_to_json(original_tensor)
        deserialized_tensor = recursive_from_json(json_data)

        self.assertTrue(torch.equal(original_tensor, deserialized_tensor))
