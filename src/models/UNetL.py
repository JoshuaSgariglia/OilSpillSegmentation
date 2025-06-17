from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization
from config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, ParametersRegistry
from dataclass import ParametersValues
from utils.misc import ParametersLoaderModel

class UNetL(ParametersLoaderModel):
    def get_parameters_values(self) -> ParametersValues:
        return self.generate_parameters_list(ParametersRegistry.UNETL)
    
    def __init__(self, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, input_channels=INP_CHANNELS):
        inputs = Input((input_height, input_width, input_channels))

        # Encoder layers
        conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)

        # Decoder layers
        deconv3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
        deconv3 = concatenate([deconv3, conv3], axis=3)
        deconv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(deconv3)
        deconv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(deconv3)

        deconv2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(deconv3)
        deconv2 = concatenate([deconv2, conv2], axis=3)
        deconv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(deconv2)
        deconv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(deconv2)

        deconv1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(deconv2)
        deconv1 = concatenate([deconv1, conv1], axis=3)
        deconv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(deconv1)
        deconv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(deconv1)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(deconv1)

        super().__init__(inputs=[inputs], outputs=[outputs], name='UNetL')