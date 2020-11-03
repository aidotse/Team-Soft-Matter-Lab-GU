import apido
import itertools
import deeptrack as dt
import numpy as np
from tensorflow.keras import layers


TEST_VARIABLES = {
    "seed": [0],
    "generator_depth": [4],
    "generator_base_breadth": [16],
    "batch_size": [8],
    "min_data_size": [200],
    "max_data_size": [400],
    "augmentation_dict": [
        {},
        {"FlipLR": {}},
        {
            "FlipLR": {},
            "Affine": {"rotate": lambda: np.random.rand() * 2 * np.pi},
        },
        {
            "FlipLR": {},
            "Affine": {
                "rotate": lambda: np.random.rand() * 2 * np.pi,
                "scale": lambda: np.random.rand() * 0.2 + 0.9,
            },
        },
        {
            "FlipLR": {},
            "Affine": {
                "rotate": lambda: np.random.rand() * 2 * np.pi,
                "shear": lambda: np.random.rand() * 0.1 - 0.05,
            },
        },
        {
            "FlipLR": {},
            "Affine": {
                "rotate": lambda: np.random.rand() * 2 * np.pi,
            },
            "ElasticTransformation": {"alpha": 70, "sigma": 7},
        },
        {
            "FlipLR": {},
            "Affine": {
                "rotate": lambda: np.random.rand() * 2 * np.pi,
                "shear": lambda: np.random.rand() * 0.1 - 0.05,
                "scale": lambda: np.random.rand() * 0.2 + 0.9,
            },
        },
        {
            "FlipLR": {},
            "Affine": {
                "rotate": lambda: np.random.rand() * 2 * np.pi,
                "shear": lambda: np.random.rand() * 0.1 - 0.05,
                "scale": lambda: np.random.rand() * 0.2 + 0.9,
            },
            "ElasticTransformation": {"alpha": 70, "sigma": 7},
        },
    ],
}


def model_initializer(generator_depth, generator_base_breadth, **kwargs):

    activation = lambda x: layers.LeakyReLU(0.2)(x)

    convolution_block = dt.layers.ConvolutionalBlock(
        activation=activation, instance_norm=True
    )
    base_block = dt.layers.ResidualBlock(activation=activation)
    pooling_block = dt.layers.ConvolutionalBlock(
        strides=2, activation=activation, instance_norm=True
    )
    deconvolution_block = dt.layers.StaticUpsampleBlock(
        kernel_size=3, instance_norm=True, activation=activation
    )

    generator = dt.models.unet(
        input_shape=(None, None, 7),  # shape of the input
        conv_layers_dimensions=list(
            generator_base_breadth * 2 ** n for n in range(generator_depth - 1)
        ),  # number of features in each convolutional layer
        base_conv_layers_dimensions=(
            generator_base_breadth * 2 ** (generator_depth - 1),
        ),  # number of features at the base of the unet
        output_conv_layers_dimensions=(
            generator_base_breadth,
            generator_base_breadth,
        ),  # number of features in convolutional layer after the U-net
        steps_per_pooling=2,  # 2                                 # number of convolutional layers per pooling layer
        number_of_outputs=3,  # number of output features
        output_activation="tanh",  # activation function on final layer
        compile=False,
        output_kernel_size=1,
        encoder_convolution_block=convolution_block,
        decoder_convolution_block=convolution_block,
        base_convolution_block=base_block,
        pooling_block=pooling_block,
        upsampling_block=deconvolution_block,
        output_convolution_block=convolution_block,
    )

    discriminator_convolution_block = dt.layers.ConvolutionalBlock(
        kernel_size=(4, 4),
        strides=2,
        activation=activation,
        instance_norm=lambda x: (
            False
            if x == 16
            else {"axis": -1, "center": False, "scale": False},
        ),
    )

    discriminator_pooling_block = lambda f: (lambda x: x)

    discriminator = dt.models.convolutional(
        input_shape=[(256, 256, 3), (256, 256, 7)],  # shape of the input
        conv_layers_dimensions=(
            16,
            32,
            64,
            128,
            256,
        ),  # number of features in each convolutional layer
        dense_layers_dimensions=(),  # number of neurons in each dense layer
        number_of_outputs=1,  # number of neurons in the final dense step (numebr of output values)
        compile=False,
        output_kernel_size=4,
        dense_top=False,
        convolution_block=discriminator_convolution_block,
        pooling_block=discriminator_pooling_block,
    )

    from tensorflow.keras.optimizers import Adam

    # model
    model = dt.models.cgan(
        generator=generator,
        discriminator=discriminator,
        discriminator_loss="mse",
        discriminator_optimizer=Adam(lr=0.0002, beta_1=0.5),
        discriminator_metrics="accuracy",
        assemble_loss=["mse", apido.combined_metric()],
        assemble_optimizer=Adam(lr=0.0002, beta_1=0.5),
    )

    return model


# Populate models
_models = []
_generators = []


def append_model(**arguments):
    _models.append((arguments, lambda: model_initializer(**arguments)))


def append_generator(**arguments):
    _generators.append(
        (
            arguments,
            lambda: apido.get_generator(
                root_path="D:/hackathon/", **arguments
            ),
        )
    )


for prod in itertools.product(*TEST_VARIABLES.values()):

    arguments = dict(zip(TEST_VARIABLES.keys(), prod))
    append_model(**arguments)
    append_generator(**arguments)


def get_model(i):
    try:
        i = int(i)
    except ValueError:
        pass

    args, model = _models[i]
    return args, model()


def get_generator(i):
    try:
        i = int(i)
    except ValueError:
        pass

    args, generator = _generators[i]
    print(i, args)
    return args, generator()
