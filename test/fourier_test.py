import unittest

import torch
from torch import tensor


class fourier_test(unittest.TestCase):
    """
    These Two Test Cases Cannot Be Both Passed On A Single Torch Environment
    ( Thanks To The Compatibility Of Torch API LOL)
    """
    def test_original_rfft_equals(self):
        """
        Run Only in an Env with torch.rfft
        Returns:

        """
        src_img = torch.load("src_img.pt")
        fft_src = torch.load("fft_src.pt")
        self.assertTrue(fft_src.equal(torch.rfft(src_img, signal_ndim=2, onesided=False)))
    def test_new_rfft_equals(self):
        """
        Run Only in Newer Version Of Torch With torch.fft update
        Returns:

        """
        src_img = torch.load("src_img.pt")
        fft_src = torch.load("fft_src.pt")
        output = torch.fft.fft2(src_img, dim=(-2, -1), s=[256, 256])
        output = torch.stack((output.real, output.imag), -1)
        print(fft_src.shape)
        print(output.shape)
        self.assertTrue(output.equal(fft_src))
        """
        By Visually Inspecting Values I think They are Roughly Equivalent.
        """

class fourier_test_2(unittest.TestCase):
    """
    These Two Test Cases Cannot Be Both Passed On A Single Torch Environment
    ( Thanks To The Compatibility Of Torch API LOL)
    """
    def test_original_rfft_equals(self):
        """
        Run Only in an Env with torch.rfft
        Returns:

        """
        src_in_trg = torch.load('src_in_trg.pt')
        fft_src_ = torch.load('fft_src_.pt')
        self.assertTrue(src_in_trg.equal(torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[256, 256])))

    def test_new_rfft_equals(self):
        """
        Run Only in Newer Version Of Torch With torch.fft update
        Returns:

        """
        src_in_trg = torch.load('src_in_trg.pt')
        fft_src_ = torch.load('fft_src_.pt')
        output_ifft_new = torch.fft.ifft2(torch.complex(fft_src_[..., 0], fft_src_[..., 1]), dim=(-2, -1))
        output_ifft_new = output_ifft_new.real
        print(src_in_trg.shape)
        print(output_ifft_new.shape)
        self.assertTrue(src_in_trg.equal(output_ifft_new))
        """
        By Visually Inspecting Values I think They are Roughly Equivalent.
        """