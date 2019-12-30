from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GRU, Bidirectional,
    Input, Dense, Activation, Reshape, BatchNormalization
)
from tensorflow_core.python.keras.layers import TimeDistributed


class OCR:
    @staticmethod
    def build(input_size, pool_size, output_size):
        conv_filters = 16
        kernel_size = (3, 3)
        time_dense_size = 32
        rnn_size = 512

        input_data = Input(name="input", shape=input_size)

        cnn = Conv2D(conv_filters, kernel_size, padding='same', kernel_initializer='he_normal')(input_data)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        cnn = MaxPooling2D(pool_size=(pool_size, pool_size))(cnn)

        cnn = Conv2D(conv_filters, kernel_size, padding='same', kernel_initializer='he_normal')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        cnn = MaxPooling2D(pool_size=(pool_size, pool_size))(cnn)

        # CNN to RNN
        shape = cnn.get_shape()
        cnn = Reshape((shape[1], shape[2] * shape[3]))(cnn)

        bgru = Bidirectional(GRU(units=rnn_size, return_sequences=True, reset_after=True, dropout=0.5))(cnn)
        bgru = TimeDistributed(Dense(units=time_dense_size))(bgru)

        bgru = Bidirectional(GRU(units=rnn_size, return_sequences=True, reset_after=True, dropout=0.5))(bgru)
        dense = TimeDistributed(Dense(units=output_size))(bgru)

        output_data = Activation("softmax", name="output")(dense)

        return input_data, output_data
