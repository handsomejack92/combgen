"""
Autoencoder model definitions

Following the Sacred configuration model using python functions, we define the
architectures of the models here. The names referece the first author of the
article from where they were take. Some might be slightly modified.

Configurations follow a general structure:
    1. gm_type (currently only lgm is available)
    2. latent_size
    3. input_size (not neccessary since it is overwritten depending on the dataset)
    4. encoder_layers: a list with layer definitions
    5. decoder layers: optional, model creation function will attemtp to transpose

Parameters in the config for each layer follow the order in Pytorch's documentation
Excluding any of them will use the default ones. We can also pass kwargs in a dict:

    ('layer_name', <list_of_args>, <dict_of_kwargs>)

This is a list of the configuration values supported:

Layer                   Paramaeters
==================================================================================
Convolution:            n-channels, size, stride, padding
Transposed Convolution: same, remeber output_padding when stride > 1! (use kwargs)
Pooling:                size, stride, padding, type
Linear:                 output size, fit bias
Flatten:                start dim, (optional, defaults=-1) end dim
Unflatten:              unflatten shape (have to pass the full shape)
Batch-norm:             dimensionality (1-2-3d)
Upsample:               upsample_shape (hard to infer automatically). Only bilinear
Non-linearity:          pass whatever arguments that non-linearity supports.
"""

def kim():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 3, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),
    ]


def abdi():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 3, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (128, 4, 2, 1)),
        ('relu',),

        ('conv', (256, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),
    ]


def montero():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 3, 64, 64

    encoder_layers = [
        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (128, 4, 2, 1)),
        ('relu',),

        ('conv', (128, 4, 2, 1)),
        ('relu',),

        ('conv', (256, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),
    ]


def watters():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 3, 64, 64

    encoder_layers = [
        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),
        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),
    ]


def sbd3():
    decoder_layers = [
        ('spatbroad', (64, 64)),

        ('conv', (64, 5, 1, 'same')),
        ('relu',),

        ('conv', (64, 5, 1, 'same')),
        ('relu',),

        ('conv', (3, 5, 1, 'same'))
    ]


def sbd4():
    decoder_layers = [
        ('spatbroad', (64, 64)),

        ('conv', (64, 5, 1, 'same')),
        ('relu',),

        ('conv', (64, 5, 1, 'same')),
        ('relu',),

        ('conv', (64, 5, 1, 'same')),
        ('relu',),

        ('conv', (3, 5, 1, 'same'))
    ]
