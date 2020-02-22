import numpy as np
import torch
import numbers
import math
from torch import nn


def gauss_kernel_1d(l, sig):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    xx = torch.range(-l // 2 + 1., l // 2 + 1., dtype=torch.float32)
    kernel = torch.exp(-xx**2 / (2. * sig**2))
    return kernel / torch.sum(kernel)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def gauss_smoothen_image(cfg, img, sigma_rel):
    fsz = cfg.pc_gauss_kernel_size
    kernel = gauss_kernel_1d(fsz, sigma_rel)
    in_channels = img.shape[-1]
    # k1 = tf.tile(tf.reshape(kernel, [1, fsz, 1, 1]), [1, 1, in_channels, 1])
    # k2 = tf.tile(tf.reshape(kernel, [fsz, 1, 1, 1]), [1, 1, in_channels, 1])
    k1 = torch.repeat_interleave(torch.reshape(kernel, [1, fsz, 1, 1]), [1, 1, in_channels, 1])
    k2 = torch.repeat_interleave(torch.reshape(kernel, [fsz, 1, 1, 1]), [1, 1, in_channels, 1])

    img_tmp = img
    img_tmp = tf.nn.depthwise_conv2d(img_tmp, k1, [1, 1, 1, 1], padding="SAME")
    img_tmp = tf.nn.depthwise_conv2d(img_tmp, k2, [1, 1, 1, 1], padding="SAME")
    return img_tmp


def separable_kernels(kernel):
    size = kernel.shape[0]
    k1 = tf.reshape(kernel, [1, 1, size, 1, 1])
    k2 = tf.reshape(kernel, [1, size, 1, 1, 1])
    k3 = tf.reshape(kernel, [size, 1, 1, 1, 1])
    return [k1, k2, k3]


def smoothing_kernel(cfg, sigma):
    fsz = cfg.pc_gauss_kernel_size
    kernel_1d = gauss_kernel_1d(fsz, sigma)
    if cfg.vox_size_z != -1:
        vox_size_z = cfg.vox_size_z
        vox_size = cfg.vox_size
        ratio = vox_size_z / vox_size
        sigma_z = sigma * ratio
        fsz_z = int(np.floor(fsz * ratio))
        if fsz_z % 2 == 0:
            fsz_z += 1
        kernel_1d_z = gauss_kernel_1d(fsz_z, sigma_z)
        k1 = tf.reshape(kernel_1d, [1, 1, fsz, 1, 1])
        k2 = tf.reshape(kernel_1d, [1, fsz, 1, 1, 1])
        k3 = tf.reshape(kernel_1d_z, [fsz_z, 1, 1, 1, 1])
        kernel = [k1, k2, k3]
    else:
        if cfg.pc_separable_gauss_filter:
            kernel = separable_kernels(kernel_1d)
    return kernel
