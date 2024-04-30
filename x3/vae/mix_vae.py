import spconv.pytorch as spconv

import x3.utils.nn as x3nn

from x3.vae.vae import V3DEncoder, V3DVAE
from x3.vae.dense_vae import DenseDecoder


class MixEncoder(V3DEncoder):
    def __init__(self, input_dim, base_channels, channels_multiple, latent_dim):
        modules = dict(
            linear1=x3nn.SparseLinear,
            gcr1=x3nn.spgcr,
            maxpool=spconv.SparseMaxPool3d,
            to_dense=spconv.ToDense,
            gcr2=x3nn.gcr,
            linear2=x3nn.Linear3D,
        )
        super(MixEncoder, self).__init__(input_dim, base_channels, channels_multiple, latent_dim, modules)


class MixVAE(V3DVAE):
    def __init__(self, input_dim, base_channels, channels_multiple, latent_dim):
        modules = dict(encoder=MixEncoder, decoder=DenseDecoder)
        super(MixVAE, self).__init__(input_dim, base_channels, channels_multiple, latent_dim, modules)
