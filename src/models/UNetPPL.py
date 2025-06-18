from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, ParametersRegistry
from dataclass import Parameters
from utils.misc import ParametersLoaderModel

class UNetPPL(ParametersLoaderModel):
    NAME = "UNetPPL"
    
    @classmethod
    def get_parameters_values(cls) -> list[Parameters]:
        return cls.generate_parameters_list(ParametersRegistry.UNETPPL)
    
    def __init__(self, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, input_channels=INP_CHANNELS):
        inputs = Input((input_height, input_width, input_channels))

        # Encoder layers
        conv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4)

        # Dense layers
        dense1_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2)
        dense1_1 = concatenate([dense1_1, conv1], axis=3)
        dense1_1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dense1_1)
        dense1_1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dense1_1)

        dense2_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3)
        dense2_1 = concatenate([dense2_1, conv2], axis=3)
        dense2_1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dense2_1)
        dense2_1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dense2_1)

        dense1_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(dense2_1)
        dense1_2 = concatenate([dense1_2, conv1, dense1_1], axis=3)
        dense1_2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dense1_2)
        dense1_2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dense1_2)

        # Decoder layers
        deconv3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
        deconv3 = concatenate([deconv3, conv3], axis=3)
        deconv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(deconv3)
        deconv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(deconv3)

        deconv2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(deconv3)
        deconv2 = concatenate([deconv2, conv2, dense2_1], axis=3)
        deconv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(deconv2)
        deconv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(deconv2)

        deconv1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(deconv2)
        deconv1 = concatenate([deconv1, conv1, dense1_1, dense1_2], axis=3)
        deconv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(deconv1)
        deconv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(deconv1)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(deconv1)

        super().__init__(self, inputs=[inputs], outputs=[outputs], name=self.NAME)