

import scipy.sparse as sp

import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

from keras import backend as K
from keras.models import load_model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
# from keras.initializers import TruncatedNormal
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger


# evaluation function
class Metrics(Callback):

    def __init__(self, train_x, train_y, val_x, val_y):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.batch_size = 2048

    # def on_batch_end(self, batch, logs=None):
    #     train_predict = np.argmax(np.asarray(self.model.predict(
    #         self.train_x[batch * self.batch_size: batch * self.batch_size + self.batch_size, :])), axis=1)
    #     train_targ = np.argmax(self.train_y[batch * self.batch_size: batch * self.batch_size + self.batch_size, :],
    #                            axis=1)
    #
    #     _train_f1 = f1_score(train_targ, train_predict, average='weighted')
    #     _train_recall = recall_score(train_targ, train_predict, average='weighted')
    #     _train_precision = precision_score(train_targ, train_predict, average='weighted')
    #     print("  train_f1: % f — train_precision: % f — train_recall % f" % (_train_f1, _train_precision, _train_recall))
    #     return

    def on_epoch_end(self, epoch, logs={}):
        print(' ')
        # train_predict = np.argmax(np.asarray(self.model.predict(self.train_x)), axis=1)
        # train_targ = np.argmax(self.train_y, axis=1)

        # _train_f1 = f1_score(train_targ, train_predict, average='micro')
        # _train_recall = recall_score(train_targ, train_predict, average='micro')
        # _train_precision = precision_score(train_targ, train_predict, average='micro')
        # print(
        #     "  train_f1: % f — train_precision: % f — train_recall % f" % (_train_f1, _train_precision, _train_recall))

        val_predict = np.argmax(np.asarray(self.model.predict(self.val_x)), axis=1)
        val_targ = np.argmax(self.val_y, axis=1)

        # _val_f1 = f1_score(val_targ, val_predict, average='micro')
        # _val_recall = recall_score(val_targ, val_predict, average='micro')
        # _val_precision = precision_score(val_targ, val_predict, average='micro')
        # print("  val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        # print("  val_f1L % f"% _val_f1)

        # rep = classification_report(val_targ, val_predict, target_names=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        rep = classification_report(val_targ, val_predict, target_names=['<= 0.2', '0.2 < & > 0.8', '>= 0.8'])

        _cm = confusion_matrix(val_targ, val_predict, labels=range(0, 3))
        print("  val_confusion_matrix, val_classfication_report: \n")
        print(_cm)
        print(rep)
        print(' ')
        return


num_class = 1

# define model
model = Sequential()

## nn
model.add(Dense(64, input_dim=419652, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(, activation='relu'))
# model.add(Dense(4, activation='relu'))
model.add(Dense(num_class, activation='sigmoid'))
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='./nn-classify-{epoch:02d}-{val_loss:.5f}.hdf5')
csv_logger = CSVLogger('./log.csv', append=True, separator=';')

train_x = sp.load_npz('./preliminary_contest_data/train_x.npz')
train_y = np.load('./preliminary_contest_data/train_y.npy')

# Y = np.array(list(map(lambda x: np.eye(num_class)[x], train_y)))

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=18)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=18)

print(y_train.shape, y_test.shape, y_val.shape)

print('x and y ready')

print('train')

# model = load_model('')

epochs = 5
batch_size = 2048
num_batch = X_train.shape[0] // batch_size

num_val_batch = X_val.shape[0] // batch_size


def train_batch_generator(X, y, batch_size, shuffle):
    number_of_batches = X.shape[0] // batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter: batch_size * (counter+1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch

        # at the end of one epoch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def evl_batch_generator(X, y, batch_size, shuffle):
    number_of_batches = X.shape[0] // batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter: batch_size * (counter+1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch

        # at the end of one epoch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


model.fit_generator(generator=train_batch_generator(X_train, y_train, batch_size=batch_size, shuffle=True),
                    validation_data=evl_batch_generator(X_val, y_val, batch_size=batch_size, shuffle=False),
                    validation_steps=num_val_batch, nb_epoch=epochs, steps_per_epoch=num_batch,
                    callbacks=[csv_logger, checkpoint])




# for e in range(epochs):
#     seed = time.time()
#     X_train, y_train = shuffle(X_train, y_train, random_state=seed)
#
#     for i, b in enumerate(range(num_batch)):
#         batch_train_x, batch_train_y = X_train[i * batch_size: (i + 1) * batch_size], \
#                                        y_train[i * batch_size: (i + 1) * batch_size]
#
#         batch_train_x, batch_train_y = batch_train_x.toarray(), batch_train_y.toarray()
#
#         model.fit(batch_train_x, batch_train_y, batch_size=batch_size,
#                   verbose=1, epochs=1, callbacks=[checkpoint])
#
#     for i, b in enumerate(range(val_num_batch)):
#         batch_val_x, batch_val_y = X_val[i * batch_size: (i + 1) * batch_size], \
#                                    y_val[i * batch_size: (i + 1) * batch_size]
#
#
#
#
#
# model.fit(X_train, y_train, batch_size=2048, verbose=2, validation_data=(X_val, y_val),
#           initial_epoch=0, epochs=5000, callbacks=[Metrics(X_train, y_train, X_val, y_val), checkpoint, csv_logger])



