import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Bidirectional,
    GRU,
)
from tensorflow.keras.optimizers import Adam

tf.keras.utils.set_random_seed(11)
tf.config.experimental.enable_op_determinism()

RNN = GRU


def create_gru_model(
    feature_dim: int,
    output_dims: int,
    units: int = 70,
    dropout: float = 0.2,
    dropout1: float = 0.2,
    dropout2: float = 0.1,
    output_activation: str = "linear",
):
    activation_func = "relu"

    encoder_inputs = layers.Input(shape=(None, feature_dim))
    encoder = GRU(units, return_state=False, return_sequences=False)
    encoder_outputs = encoder(encoder_inputs)
    encoder_outputs = Dropout(dropout)(encoder_outputs)

    # Dense layers
    outputs = Dense(int(units / 2), activation=activation_func)(encoder_outputs)
    outputs = Dropout(dropout1)(outputs)
    outputs = Dense(int(units / 4), activation=activation_func)(outputs)
    outputs = Dropout(dropout2)(outputs)
    outputs = Dense(output_dims, activation=output_activation)(outputs)
    return keras.Model(encoder_inputs, outputs)


def create_lasenet(
    feature_dim: int,
    discrete_output_dim: int,
    continuous_output_dim: int,
    units: int = 70,
    dropout: float = 0.2,
    dropout1: float = 0.2,
    dropout2: float = 0.1,
):
    activation_func = "relu"
    # Bidirectional
    encoder_inputs = layers.Input(shape=(None, feature_dim))
    encoder = Bidirectional(RNN(units, return_state=True, return_sequences=True))
    encoder_outputs, _, _ = encoder(encoder_inputs)
    encoder_outputs = Dropout(dropout)(encoder_outputs)

    d_outputs_1 = Dense(int(units / 2), activation=activation_func)(encoder_outputs)
    d_outputs_1 = Dropout(dropout1)(d_outputs_1)
    d_outputs_1 = Dense(int(units / 4), activation=activation_func)(d_outputs_1)
    d_outputs_1 = Dropout(dropout2)(d_outputs_1)

    outputs = []
    if discrete_output_dim > 0:
        k_outputs = Dense(
            discrete_output_dim, activation="softmax", name="discrete_latent"
        )(d_outputs_1)
        outputs.append(k_outputs)

    if continuous_output_dim > 0:
        linear_outputs = Dense(
            continuous_output_dim, activation="linear", name="continuous_latent"
        )(d_outputs_1)
        outputs.append(linear_outputs)

    return keras.Model(inputs=encoder_inputs, outputs=outputs)


def get_model_for_parameter_estimation(params):
    model = create_gru_model(
        params["feature_dim"],
        params["output_dims"],
        params["units"],
        params["dropout"],
        params["dropout1"],
        params["dropout2"],
    )
    optimizer = Adam(learning_rate=params["learning_rate"])
    model.compile(
        loss="mse",
        optimizer=optimizer,
    )
    return model

def get_model_for_model_identification(params):
    model = create_gru_model(
        params["feature_dim"],
        params["output_dims"],
        params["units"],
        params["dropout"],
        params["dropout1"],
        params["dropout2"],
        output_activation="softmax",
    )
    optimizer = Adam(learning_rate=params["learning_rate"])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model

def get_lasenet_for_continuous_output(params):
    lasenet = create_lasenet(
        params["feature_dim"],
        params["discrete_output_dim"],
        params["continuous_output_dim"],
        params["units"],
        params["dropout"],
        params["dropout1"],
        params["dropout2"],
    )
    optimizer = Adam(learning_rate=params["learning_rate"])
    lasenet.compile(
        loss={
            "continuous_latent": "mse",
        },
        loss_weights={"continuous_latent": 1},
        optimizer=optimizer,
    )
    return lasenet
