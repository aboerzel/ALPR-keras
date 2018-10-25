import itertools
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
from config import anpr_config as config


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode(out):
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(config.ALPHABET):
                outstr += config.ALPHABET[c]
        return outstr


sess = tf.Session()
K.set_session(sess)

model = load_model(config.MODEL_PATH, compile=False)

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

net_inp = model.get_layer(name='input').input
net_out = model.get_layer(name='softmax').output

img_filepath = 'D:/development/cv/datasets/anpr/test/ROL-L716.png'
label = img_filepath.split('/')[-1].split('.')[0]

stream = open(img_filepath, "rb")
bytes = bytearray(stream.read())
numpyarray = np.asarray(bytes, dtype=np.uint8)
img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (160, 32))
img = img.astype(np.float32)
img /= 255

#plt.imshow(img, cmap='gray')
#plt.show()

img = np.expand_dims(img.T, -1)
X_data = [img]

# batch_output = model.predict(X_data)
net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})
pred_text = decode(net_out_value)
fig = plt.figure(figsize=(10, 10))
outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
ax1 = plt.Subplot(fig, outer[0])
fig.add_subplot(ax1)
ax2 = plt.Subplot(fig, outer[1])
fig.add_subplot(ax2)
print('Predicted: %s\nTrue: %s' % (pred_text, label))
img = X_data[0][:, :, 0].T
ax1.set_title('Input img')
ax1.imshow(img, cmap='gray')
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_title('Activations')
ax2.imshow(net_out_value[0].T, cmap='binary', interpolation='nearest')
ax2.set_yticks(list(range(len(config.ALPHABET) + 1)))
ax2.set_yticklabels(config.ALPHABET) # + ['blank'])
ax2.grid(False)
for h in np.arange(-0.5, len(config.ALPHABET) + 1 + 0.5, 1):
    ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)

plt.show()

