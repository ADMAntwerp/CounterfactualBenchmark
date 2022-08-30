from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

def nn_ohe(input_shape, hidden_layers_ws, output_number):
    x_in = Input(shape=(input_shape,))
    x = Dense(hidden_layers_ws, activation='relu')(x_in)
    x_out = Dense(2, activation='softmax')(x)
    if output_number == 1:
        x_bin = Dense(1, activation='linear')(x_out)
        nn = Model(inputs=x_in, outputs=x_bin)
    if output_number == 2:
        nn = Model(inputs=x_in, outputs=x_out)

    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return nn