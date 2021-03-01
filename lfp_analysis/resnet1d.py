from fastai.vision.all import *


def _conv_block1d(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni, nf // 4, 1, ndim=1),
        ConvLayer(nf // 4, nf // 4, stride=stride, ndim=1),
        ConvLayer(nf // 4, nf, 1, act_cls=None, norm_type=NormType.BatchZero, ndim=1),
    )


class ResBlock1d(Module):
    def __init__(self, ni, nf, stride=1):
        self.convs = _conv_block1d(ni, nf, stride)
        self.idconv = noop if ni == nf else ConvLayer(ni, nf, 1, act_cls=None, ndim=1)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


def _resnet_stem1d(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i + 1], 3, stride=2 if i == 0 else 1, ndim=1)
        for i in range(len(sizes) - 1)
    ] + [nn.MaxPool1d(kernel_size=3, stride=2, padding=1)]


class ResNet1d(nn.Sequential):
    def __init__(self, n_in, n_out, layers, expansion=1):
        stem = _resnet_stem1d(n_in, 32, 32, 64)
        self.block_szs = [64, 64, 128, 256, 512]
        for i in range(1, 5):
            self.block_szs[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]
        super().__init__(
            *stem,
            *blocks,
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(self.block_szs[-1], n_out),
            Flatten()
        )

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx == 0 else 2
        ch_in, ch_out = self.block_szs[idx : idx + 2]
        return nn.Sequential(
            *[
                ResBlock1d(ch_in if i == 0 else ch_out, ch_out, stride if i == 0 else 1)
                for i in range(n_layers)
            ]
        )
