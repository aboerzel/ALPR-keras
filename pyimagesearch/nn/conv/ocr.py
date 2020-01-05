from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, LSTM, GRU, Bidirectional,
    Input, Dense, Activation, Reshape, BatchNormalization, add, concatenate
)


class OCR:
    @staticmethod
    def conv_bgru(input_shape, output_size):
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        time_dense_size = 32
        rnn_size = 512

        input_data = Input(name="input", shape=input_shape)

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
        dense = Dense(time_dense_size, activation='relu', kernel_initializer='he_normal')(cnn)

        # RNN layer
        bgru = Bidirectional(GRU(units=rnn_size, return_sequences=True), merge_mode="sum")(dense)
        bgru = BatchNormalization()(bgru)
        bgru = Bidirectional(GRU(units=rnn_size, return_sequences=True), merge_mode="concat")(bgru)
        bgru = BatchNormalization()(bgru)

        # transforms RNN output to character activations:
        dense = Dense(output_size, kernel_initializer='he_normal')(bgru)
        output_data = Activation("softmax", name="output")(dense)

        return input_data, output_data

    @staticmethod
    def conv_blstm(input_shape, output_size):
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        time_dense_size = 32
        rnn_size = 512

        input_data = Input(name="input", shape=input_shape)

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
        dense = Dense(time_dense_size, activation='relu', kernel_initializer='he_normal')(cnn)

        # RNN layer
        bgru = Bidirectional(LSTM(units=rnn_size, return_sequences=True), merge_mode="sum")(dense)
        #bgru = BatchNormalization()(bgru)
        bgru = Bidirectional(LSTM(units=rnn_size, return_sequences=True), merge_mode="concat")(bgru)
        #bgru = BatchNormalization()(bgru)

        # transforms RNN output to character activations:
        dense = Dense(output_size, kernel_initializer='he_normal')(bgru)
        output_data = Activation("softmax", name="output")(dense)

        return input_data, output_data

    @staticmethod
    def vgg_blstm(input_shape, output_size):

        input_data = Input(name='input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)
        cnn = OCR.__vgg__(input_data)

        # CNN to RNN
        cnn = Reshape(target_shape=(32, 2048))(cnn)  # (None, 32, 2048)
        dense = Dense(64, activation='relu', kernel_initializer='he_normal')(cnn)  # (None, 32, 64)

        # RNN layer
        lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal')(dense)  # (None, 32, 512)
        lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(cnn)
        lstm1_merged = add([lstm_1, lstm_1b])  # (None, 32, 512)
        lstm1_merged = BatchNormalization()(lstm1_merged)
        lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal')(lstm1_merged)
        lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(lstm1_merged)
        lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, 32, 1024)
        lstm2_merged = BatchNormalization()(lstm2_merged)

        # transforms RNN output to character activations:
        dense = Dense(output_size, kernel_initializer='he_normal')(lstm2_merged)  # (None, 32, 42)
        output_data = Activation('softmax', name='output')(dense)

        return input_data, output_data

    @staticmethod
    def vgg_bgru(input_shape, output_size):

        input_data = Input(name='input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)
        cnn = OCR.__vgg__(input_data)

        # CNN to RNN
        shape = cnn.get_shape()
        cnn = Reshape((shape[1], shape[2] * shape[3]))(cnn)
        dense = Dense(64, activation='relu', kernel_initializer='he_normal')(cnn)  # (None, 32, 64)

        # RNN layer
        bgru = Bidirectional(GRU(units=256, return_sequences=True, dropout=0.2), merge_mode="sum")(dense)
        bgru = BatchNormalization()(bgru)
        bgru = Bidirectional(GRU(units=256, return_sequences=True, dropout=0.2), merge_mode="concat")(bgru)
        bgru = BatchNormalization()(bgru)

        # transforms RNN output to character activations:
        dense = Dense(output_size, kernel_initializer='he_normal')(bgru)  # (None, 32, 42)
        output_data = Activation('softmax', name='output')(dense)

        return input_data, output_data

    @staticmethod
    def __vgg__(input_data):

        cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(input_data)
        # cnn = BatchNormalization()(cnn)
        cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn)
        # cnn = BatchNormalization()(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

        cnn = Conv2D(64, (3, 3), activation='relu', padding='same')(cnn)
        # cnn = BatchNormalization()(cnn)
        cnn = Conv2D(64, (3, 3), activation='relu', padding='same')(cnn)
        # cnn = BatchNormalization()(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

        return cnn

    @staticmethod
    def __vgg2__(input_data):
        # convolution layer with kernel size (3,3)
        cnn = Conv2D(64, (3, 3), activation='relu', padding='same')(input_data)
        cnn = MaxPooling2D(pool_size=(2, 2), strides=2)(cnn)

        cnn = Conv2D(128, (3, 3), activation='relu', padding='same')(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2), strides=2)(cnn)

        cnn = Conv2D(256, (3, 3), activation='relu', padding='same')(cnn)

        cnn = Conv2D(256, (3, 3), activation='relu', padding='same')(cnn)
        cnn = MaxPooling2D(pool_size=(1, 2))(cnn)

        cnn = Conv2D(512, (3, 3), activation='relu', padding='same')(cnn)
        cnn = BatchNormalization()(cnn)

        cnn = Conv2D(512, (3, 3), activation='relu', padding='same')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling2D(pool_size=(1, 2))(cnn)

        cnn = Conv2D(512, (2, 2), activation='relu', padding='same')(cnn)

        return cnn
