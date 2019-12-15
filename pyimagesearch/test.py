import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Add, Lambda
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras import optimizers

from scipy.io import loadmat
from scipy.spatial.distance import cdist

tf.compat.v1.disable_eager_execution()

x = np.random.rand(100).reshape(100, 1).astype(float)
y = np.array(x > 0.5).reshape(100, 1).astype(float)

input1 = Input(shape=(1,), name='input1')
d1 = Dense(12, name='d11')(input1)
d1 = Dense(12, name='d12')(d1)
d1 = Dense(1, name='output1')(d1)

input2 = Input(shape=(1,), name='input2')
d2 = Dense(12, name='d21')(input2)
d2 = Dense(12, name='d22')(d2)
d2 = Dense(1, name='output2')(d2)

label = Input(shape=(1,), name='label')

model1 = Model(inputs=[input1, label], outputs=d1)
model2 = Model(inputs=input2, outputs=d2)
model2_cl = clone_model(model2)
model1_cl = clone_model(model1)

pre_result1 = model1.predict({'input1': x, 'label': y})
pre_result1_cl = model1_cl.predict({'input1': x, 'label': y})
pre_result2 = model2.predict(x)


def cust_loss(xi, yi, yp):
    loss = tf.reduce_mean((yi - yp) ** 2) + tf.reduce_mean((model2(xi) - yi) ** 2)

    return loss


model1.add_loss(cust_loss(input1, label, d1))
# optimizer = optimizers.Adam(lr = 3e-4)
optimizer = tf.compat.v1.train.AdamOptimizer(3e-4)
model1.compile(optimizer, loss=None)

model1.fit({'input1': x, 'label': y}, None, epochs=100)
post_result1 = model1.predict({'input1': x, 'label': y})
post_result1_cl = model1_cl.predict({'input1': x, 'label': y})
post_result2 = model2.predict(x)

print(sum(pre_result2 != post_result2))