from tensorflow.keras import backend as K, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Dense, GRU, add, concatenate, Activation, Lambda


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class OCR:
    @staticmethod
    def build(width, height, pool_size, output_size, max_text_len):
        conv_filters = 16
        kernel_size = (3, 3)
        time_dense_size = 32
        rnn_size = 512

        input_shape = (width, height, 1)

        act = 'relu'
        input_data = Input(name='input', shape=input_shape, dtype='float32')

        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

        conv_to_rnn_dims = (width // (pool_size ** 2), (height // (pool_size ** 2)) * conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

        # Two layers of bidirecitonal GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                     name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                     name='gru2_b')(gru1_merged)

        # transforms RNN output to character activations:
        inner = Dense(output_size, kernel_initializer='he_normal', name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)
        Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='labels', shape=[max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        return Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
