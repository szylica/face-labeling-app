import os
from keras.applications import VGG16
from keras import models, layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

# Ścieżki do katalogów
train_dir = "C:\Users\nieza\OneDrive\Pulpit\datasets_emotion_generator-main\emotions\trening"
val_dir = "C:\Users\nieza\OneDrive\Pulpit\datasets_emotion_generator-main\emotions\walidacja"
test_dir = "C:\Users\nieza\OneDrive\Pulpit\datasets_emotion_generator-main\emotions\test"

# Parametry
img_height, img_width = 150, 150  # Wymagane przez VGG16
batch_size = 64

# Przygotowanie generatorów danych
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

# Wczytanie modelu VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,  # Usuwamy warstwy w pełni połączone
                  input_shape=(img_height, img_width, 3))

# Zamrożenie wag modelu bazowego
conv_base.trainable = False

# Stworzenie modelu
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))  # Regularizacja
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))  # Wyjście wieloklasowe

# Kompilacja modelu
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Trenowanie modelu
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr, tensorboard_callback],
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size
)

# Zapis modelu
model.save("emotion_detection_vgg16.h5")

