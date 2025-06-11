from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

def unet_plus_plus(nClasses, input_height, input_width):
    inputs = Input((input_height, input_width, 3))

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    # Decoder levels
    # Level 0 (up6): resolution H/8, W/8
    conv6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv5)
    merge1 = concatenate([conv6, conv4], axis=-1)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(merge1)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    # Level 1 (up7): resolution H/4, W/4
    conv7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv6)
    merge2 = concatenate([conv7, conv3], axis=-1)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(merge2)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    # Level 2 (up8): resolution H/2, W/2
    conv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7)
    merge3 = concatenate([conv8, conv2], axis=-1)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(merge3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    # Level 3 (up9): resolution H, W
    conv9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8)
    merge4 = concatenate([conv9, conv1], axis=-1)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(merge4)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    # Output layer
    conv10 = Conv2D(nClasses, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model

if __name__ == '__main__':
    m = unet_plus_plus(5, 256, 256)
    m.summary()
