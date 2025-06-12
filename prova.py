import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import time
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.config.list_physical_devices('GPU'))

def training():
    # Load data
    dataset_name = 'cifar10'
    (train_dataset, test_dataset), dataset_info = tfds.load(name=dataset_name,
                                                split=['train', 'test'],
                                                shuffle_files=True,
                                                with_info=True,
                                                as_supervised=True)
    
    # Get the number of classes in the dataset
    num_classes = dataset_info.features['label'].num_classes
    num_classes

    # Preprocess the data
    def preprocess_data(image, label):
        # Convert image to float32 and normalize between 0 and 1
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Apply preprocessing to the datasets
    train_dataset = train_dataset.map(preprocess_data)
    test_dataset = test_dataset.map(preprocess_data)

    # Step 3: Build the model
    input_dim = (32, 32, 3)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    #Compile the model
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    batch_size = 128
    num_epochs = 10
    # To process the dataset in batches create the batches of batch_size
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)

    #Train the model
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

    # Step 7: Evaluate the model
    loss, accuracy = model.evaluate(test_dataset)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

with tf.device('/GPU:0'):
    start = time.time()
    training()
    print(f"GPU: {time.time() - start} seconds")

'''
# Twice as slow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    start = time.time()
    training()
    print(f"Dual GPU: {time.time() - start} seconds")
'''