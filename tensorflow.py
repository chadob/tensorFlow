# Used chatgpt as a resource to figure out how the data should be formatted.
# It also helped with understanding the padding='same'

import tensorflow as tf

mnist = tf.keras.datasets.mnist

#loads the data from the mnist data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizes the data to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    # add a convolutional layer with 6 filters and 5x5 kernels
    # uses padding = same to keep images the same size
    # uses the sigmoid function as the activation function
    tf.keras.layers.Conv2D(6, 5, activation='sigmoid', padding='same', input_shape=(28, 28, 1)),
    # pools with a size of 2 and stride of 2
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
    # 16 filters 
    tf.keras.layers.Conv2D(16, 5, activation='sigmoid', padding='same'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
    # flattens data for the output neurons
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)