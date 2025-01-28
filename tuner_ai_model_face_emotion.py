import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

# Define paths
train_dir = "emotions/trening"
val_dir = "emotions/walidacja"
test_dir = "emotions/test"
model_path = "models/emotion_detection_model.h5"

img_height, img_width = 96, 96
batch_size = 64


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


def build_model(hp):
    # Hyperparameters to tune
    conv_layers = hp.Int('conv_layers', min_value=2, max_value=5, default=4)
    initial_filters = hp.Int('initial_filters', min_value=16, max_value=64, step=16)
    dense_layers = hp.Int('dense_layers', min_value=1, max_value=3)
    dense_neurons = hp.Int('dense_neurons', min_value=128, max_value=512, step=128)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd'])
    activation = hp.Choice('activation', values=['relu', 'leaky_relu', 'tanh'])

    model = Sequential()

    # First Conv2D layer
    model.add(Conv2D(initial_filters, (3, 3), activation=activation, 
                     input_shape=(img_height, img_width, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Additional Conv2D layers
    filters = initial_filters
    for i in range(conv_layers - 1):
        filters *= 2
        model.add(Conv2D(filters, (3, 3), activation=activation))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Dense layers
    for _ in range(dense_layers):
        model.add(Dense(dense_neurons, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(train_generator.num_classes, activation='softmax'))

    # Optimizer selection
    if optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, **kwargs)

# Create the tuner
tuner = kt.BayesianOptimization(
    MyClassificationHyperModel(),
    objective='val_accuracy',
    max_trials=10,  # Liczba prób do przetestowania
    seed=42,
    directory='model_tuning',
    project_name='cnn_tuning'
)

# Callbacks for training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

log_dir = "model_tuning/cnn_tuning" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False)

# Search for the best hyperparameters
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stopping, reduce_lr, tensorboard_callback]
)

best_hps = tuner.get
