from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    Bidirectional,
    GRU,
    Concatenate,
)
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD

def create_multiple_q_values_model(
    feature_dim: int,
    units: int = 70,
    dropout: float = 0.2,
    dropout1: float = 0.2,
    dropout2: float = 0.1,
    learning_rate: float = 3e-4,
    cue_output_dim: int = 2,
    output_dim: int = 1,
):
    activation_func = "relu"
    # Bidirectional
    encoder_inputs = layers.Input(shape=(None, feature_dim))
    encoder = Bidirectional(GRU(units, return_state=False, return_sequences=True))
    encoder_outputs = encoder(encoder_inputs)
    encoder_outputs = Dropout(dropout)(encoder_outputs) #Dropout(dropout)(encoder_outputs)

    d_outputs_1 = Dense(int(units / 2), activation=activation_func)(encoder_outputs)
    d_outputs_1 = Dropout(dropout1)(d_outputs_1)
    d_outputs_1 = Dense(int(units / 4), activation=activation_func)(d_outputs_1)
    d_outputs_1 = Dropout(dropout2)(d_outputs_1)

    k_outputs = Dense(cue_output_dim, activation="softmax", name="chosen_cue")(d_outputs_1)
    # Dense layers
    ql_outputs = Dense(output_dim, activation="linear", name="q_left")(
        d_outputs_1
    )
    # Dense layers
    qr_outputs = Dense(output_dim, activation="linear", name="q_right")(
        d_outputs_1
    )

    model = keras.Model(inputs=encoder_inputs, outputs=[k_outputs, ql_outputs, qr_outputs])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss={
            "chosen_cue": "categorical_crossentropy",
            "q_left": "mse",
            "q_right": "mse",
        },
        loss_weights={"chosen_cue": 0.2, "q_left": 1,  "q_right": 1},
        optimizer=optimizer,
    )
    return model