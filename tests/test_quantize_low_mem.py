import importlib
import os
import tempfile
import unittest


import torch

class LowMemoryQuantizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quantize = importlib.import_module(
            "exllamav3.modules.quant.exl3_lib.quantize"
        )

    def test_mmap_zeros_is_writable_and_leaves_no_named_tempfile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tensor = self.quantize._mmap_zeros(
                (8, 16), torch.float32, temp_dir
            )
            self.assertEqual(tensor.shape, (8, 16))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(torch.count_nonzero(tensor).item(), 0)
            tensor[3, 4] = 7.0
            self.assertEqual(tensor[3, 4].item(), 7.0)
            self.assertEqual(os.listdir(temp_dir), [])

    def test_streamed_pair_trace_matches_dense_reference(self):
        generator = torch.Generator().manual_seed(17)
        weight = torch.randn(8, 13, generator=generator)
        quantized = weight + 0.05 * torch.randn(8, 13, generator=generator)
        matrix = torch.randn(8, 8, generator=generator)
        hessian = matrix.T @ matrix + torch.eye(8)
        expected_num = torch.einsum(
            "ik,ij,jk->", weight - quantized, hessian, weight - quantized
        ).item()
        expected_den = torch.einsum(
            "ik,ij,jk->", weight, hessian, weight
        ).item()
        actual_num, actual_den = self.quantize.block_trace_pair_stream(
            weight,
            quantized,
            hessian,
            torch.device("cpu"),
            col_block=3,
        )
        self.assertAlmostEqual(actual_num, expected_num, places=4)
        self.assertAlmostEqual(actual_den, expected_den, places=4)


if __name__ == "__main__":
    unittest.main()
