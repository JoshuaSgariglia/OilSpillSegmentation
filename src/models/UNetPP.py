
from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

from Config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH

def UNetPP(input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, input_channels=INP_CHANNELS):
    
    inputs = Input((input_height, input_width, input_channels))

    # Encoder layers
    conv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    conv1 = Dropout(0.5) (conv1)
    conv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1)
    conv1 = Dropout(0.5) (conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2)) (conv1)

    conv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool1)
    conv2 = Dropout(0.5) (conv2)
    conv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2)
    conv2 = Dropout(0.5) (conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2)) (conv2)

    conv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool2)
    conv3 = Dropout(0.5) (conv3)
    conv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3)
    conv3 = Dropout(0.5) (conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool3)
    conv4 = Dropout(0.5) (conv4)
    conv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4)
    conv4 = Dropout(0.5) (conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool4)
    conv5 = Dropout(0.5) (conv5)
    conv5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv5)
    conv5 = Dropout(0.5) (conv5)

    # Dense layers (first row)
    dense1_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2)
    dense1_1 = concatenate([dense1_1, conv1], axis=3)
    dense1_1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_1)
    dense1_1 = Dropout(0.5) (dense1_1)
    dense1_1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_1)
    dense1_1 = Dropout(0.5) (dense1_1)

    dense1_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(dense2_1)
    dense1_2 = concatenate([dense1_2, conv1, dense1_1], axis=3)
    dense1_2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_2)
    dense1_2 = Dropout(0.5) (dense1_2)
    dense1_2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_2)
    dense1_2 = Dropout(0.5) (dense1_2)
    
    dense1_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(dense2_2)
    dense1_3 = concatenate([dense1_3, conv1, dense1_1, dense1_2],  axis=3)
    dense1_3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_3)
    dense1_3 = Dropout(0.5) (dense1_3)
    dense1_3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_3)
    dense1_3 = Dropout(0.5) (dense1_3)

    # Dense layers (second row)
    dense2_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3)
    dense2_1 = concatenate([dense2_1, conv2],  axis=3)
    dense2_1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense2_1)
    dense2_1 = Dropout(0.5) (dense2_1)
    dense2_1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense2_1)
    dense2_1 = Dropout(0.5) (dense2_1)

    dense2_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(dense3_1)
    dense2_2 = concatenate([dense2_2, conv2, dense2_1], axis=3)
    dense2_2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense2_2)
    dense2_2 = Dropout(0.5) (dense2_2)
    dense2_2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense2_2)
    dense2_2 = Dropout(0.5) (dense2_2)

    # Dense layers (third row)
    dense3_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    dense3_1 = concatenate([dense3_1, conv3], axis=3)
    dense3_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense3_1)
    dense3_1 = Dropout(0.5) (dense3_1)
    dense3_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense3_1)
    dense3_1 = Dropout(0.5) (dense3_1)

    # Decoder layers
    deconv4 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    deconv4 = concatenate([deconv4, conv4], axis=3)
    deconv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv4)
    deconv4 = Dropout(0.5) (deconv4)
    deconv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv4)
    deconv4 = Dropout(0.5) (deconv4)

    deconv3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(deconv4)
    deconv3 = concatenate([deconv3, conv3, dense3_1], axis=3)
    deconv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv3)
    deconv3 = Dropout(0.5) (deconv3)
    deconv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv3)
    deconv3 = Dropout(0.5) (deconv3)

    deconv2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(deconv3)
    deconv2 = concatenate([deconv2, conv2, dense2_1, dense2_2], axis=3)
    deconv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv2)
    deconv2 = Dropout(0.5) (deconv2)
    deconv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv2)
    deconv2 = Dropout(0.5) (deconv2)

    deconv1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(deconv2)
    deconv1 = concatenate([deconv1, conv1, dense1_1, dense1_2, dense1_3], axis=3)
    deconv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv1)
    deconv1 = Dropout(0.5) (deconv1)
    deconv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv1)
    deconv1 = Dropout(0.5) (deconv1)

    unetpp_output = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(deconv1)

    model = Model([inputs], [unetpp_output], 'UNetPP')
    return model


if __name__ == '__main__':
    m = UNetPP(256, 256)
    m.summary()