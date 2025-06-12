
from keras import Model
from keras.applications import vgg16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, concatenate, UpSampling2D
from keras.layers import MaxPooling2D

def UNet(nClasses, input_height, input_width):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    vgg_streamlined = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)
    assert isinstance(vgg_streamlined, Model)

    # Decoder layers
    o = UpSampling2D((2, 2))(vgg_streamlined.output)
    o = concatenate([vgg_streamlined.get_layer(
        name="block4_pool").output, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block3_pool").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block2_pool").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block1_pool").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    # The UNet network performs a 2x mirror upsampling on the input, so the final input and output are reduced by half.
    # Here, directly upsample to restore the original size.
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = Conv2D(nClasses, (1, 1), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)

    o = Reshape((-1, nClasses))(o)
    o = Activation("sigmoid")(o)

    model = Model(inputs=img_input, outputs=o)
    return model

def unet_fengfan(nClasses, input_height, input_width):
    inputs = Input((input_height, input_width, 3))

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

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(nClasses, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


if __name__ == '__main__':
    m = unet_fengfan(5, 256, 256)
    # print(m.get_weights()[2]) # Check if the weights have changed, used for testing VGG weight loading
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_unet.png')
    print(len(m.layers))
    m.summary()
