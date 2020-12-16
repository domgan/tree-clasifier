from tensorflow import keras
from preprocessing import all_images_shuffled, all_labels_shuffled

filters = 64
model = keras.Sequential([
    keras.layers.Conv2D(filters, (5, 5), padding='same', input_shape=all_images_shuffled.shape[1:], activation='relu'),
    keras.layers.Conv2D(filters, (5, 5), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters*2, (3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(filters*2, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(filters*4, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(all_images_shuffled, all_labels_shuffled, epochs=5)

# results = model.evaluate(Vowel_Feature_Test.test_input, Vowel_Feature_Test.test_labels)
# print('test loss, test acc:', results)

# model.save('vowel_model.h5', include_optimizer=False)
# print("Saved model to disk")