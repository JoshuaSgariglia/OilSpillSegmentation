from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

from config import DROPOUT_RATE, INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH

def UNetPPL(input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, input_channels=INP_CHANNELS, dropout_rate=DROPOUT_RATE):
    
    inputs = Input((input_height, input_width, input_channels))

    # Encoder layers
    conv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    conv1 = Dropout(dropout_rate) (conv1)
    conv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1)
    conv1 = Dropout(dropout_rate) (conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2)) (conv1)

    conv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool1)
    conv2 = Dropout(dropout_rate) (conv2)
    conv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2)
    conv2 = Dropout(dropout_rate) (conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2)) (conv2)

    conv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool2)
    conv3 = Dropout(dropout_rate) (conv3)
    conv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3)
    conv3 = Dropout(dropout_rate) (conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool3)
    conv4 = Dropout(dropout_rate) (conv4)
    conv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4)
    conv4 = Dropout(dropout_rate) (conv4)

    # Dense layers (first row)
    dense1_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2)
    dense1_1 = concatenate([dense1_1, conv2], axis=3)
    dense1_1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_1)
    dense1_1 = Dropout(dropout_rate) (dense1_1)
    dense1_1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_1)
    dense1_1 = Dropout(dropout_rate) (dense1_1)

    dense1_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(dense2_1)
    dense1_2 = concatenate([dense1_2, conv2, dense1_1], axis=3)
    dense1_2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_2)
    dense1_2 = Dropout(dropout_rate) (dense1_2)
    dense1_2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense1_2)
    dense1_2 = Dropout(dropout_rate) (dense1_2)

    # Dense layers (second row)
    dense2_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
    dense2_1 = concatenate([dense2_1, conv3], axis=3)
    dense2_1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense2_1)
    dense2_1 = Dropout(dropout_rate) (dense2_1)
    dense2_1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (dense2_1)
    dense2_1 = Dropout(dropout_rate) (dense2_1)

    # Decoder layers
    deconv3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    deconv3 = concatenate([deconv3, conv3], axis=3)
    deconv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv3)
    deconv3 = Dropout(dropout_rate) (deconv3)
    deconv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv3)
    deconv3 = Dropout(dropout_rate) (deconv3)

    deconv2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(deconv3)
    deconv2 = concatenate([deconv2, conv2, dense2_1], axis=3)
    deconv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv2)
    deconv2 = Dropout(dropout_rate) (deconv2)
    deconv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv2)
    deconv2 = Dropout(dropout_rate) (deconv2)

    deconv1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(deconv2)
    deconv1 = concatenate([deconv1, conv1, dense1_1, dense1_2], axis=3)
    deconv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv1)
    deconv1 = Dropout(dropout_rate) (deconv1)
    deconv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (deconv1)
    deconv1 = Dropout(dropout_rate) (deconv1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(deconv1)

    model = Model(inputs=[inputs], outputs=[outputs], name='UNetPPL')
    return model

if __name__ == '__main__':
    m = UNetPPL(256, 256)
    m.summary()