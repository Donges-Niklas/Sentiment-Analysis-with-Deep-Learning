import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i , sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.array(train_labels).astype("float32")
y_test = np.array(test_labels).astype("float32")

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


model = models.Sequential()

# Input-Layer
model.add(layers.Dense(16,activation="relu",input_shape=(10000,)))

# Hidden-Layers
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(55,activation="relu"))
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
model.add(layers.Dense(85,activation="relu"))
model.add(layers.Dense(85,activation="relu"))
model.add(layers.Dense(85,activation="relu"))
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
model.add(layers.Dense(55,activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(16,activation="relu"))

# Output-Layer
model.add(layers.Dense(46, activation="softmax"))

model.summary()

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"] 
)

x_val = x_train[:4000]
partial_x_train = x_train[4000:]
y_val = y_train[:4000]
partial_y_train = y_train[4000:]

history = model.fit(
partial_x_train, partial_y_train,
epochs=10,
batch_size=512,
validation_data=(x_val,y_val)
)
history_dict = history.history
val_acc_values = history_dict["val_acc"]

print("Validation Accuracy:", np.mean(val_acc_values))