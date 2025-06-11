from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

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
    decoder_outputs = []

    # Level 0 (up6): resolution H/8, W/8
    conv5_up = UpSampling2D(size=(2, 2))(conv5)
    merge1 = concatenate([conv5_up, conv4], axis=-1)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(merge1)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
    decoder_outputs.append(conv6)

    # Level 1 (up7): resolution H/4, W/4
    conv5_up2 = UpSampling2D(size=(4, 4))(conv5)
    conv4_up = UpSampling2D(size=(2, 2))(conv4)
    conv6_up = UpSampling2D(size=(2, 2))(conv6)
    merge2 = concatenate([conv5_up2, conv4_up, conv3, conv6_up], axis=-1)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(merge2)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)
    decoder_outputs.append(conv7)

    # Level 2 (up8): resolution H/2, W/2
    conv5_up3 = UpSampling2D(size=(8, 8))(conv5)
    conv4_up2 = UpSampling2D(size=(4, 4))(conv4)
    conv3_up = UpSampling2D(size=(2, 2))(conv3)
    conv6_up2 = UpSampling2D(size=(4, 4))(conv6)
    conv7_up = UpSampling2D(size=(2, 2))(conv7)
    merge3 = concatenate([conv5_up3, conv4_up2, conv3_up, conv2, conv6_up2, conv7_up], axis=-1)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(merge3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)
    decoder_outputs.append(conv8)

    # Level 3 (up9): resolution H, W
    conv5_up4 = UpSampling2D(size=(16, 16))(conv5)
    conv4_up3 = UpSampling2D(size=(8, 8))(conv4)
    conv3_up2 = UpSampling2D(size=(4, 4))(conv3)
    conv2_up = UpSampling2D(size=(2, 2))(conv2)
    conv6_up3 = UpSampling2D(size=(8, 8))(conv6)
    conv7_up2 = UpSampling2D(size=(4, 4))(conv7)
    conv8_up = UpSampling2D(size=(2, 2))(conv8)
    merge4 = concatenate([conv5_up4, conv4_up3, conv3_up2, conv2_up, conv1, conv6_up3, conv7_up2, conv8_up], axis=-1)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(merge4)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    # Output layer
    conv10 = Conv2D(nClasses, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model

if __name__ == '__main__':
    m = unet_plus_plus(5, 256, 256)
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_unet_plus_plus.png')
    m.summary()
