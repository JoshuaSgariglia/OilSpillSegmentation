import tensorflow as tf
from tensorflow.keras import layers, Sequential # type: ignore

from config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS, ParametersRegistry
from dataclass import Parameters
from utils.misc import ParametersLoaderModel # type: ignore

def Interpolation(filters):
    """
    Upscales H and W by 2x using bilinear interpolation,
    and reduces the channel dimension by 4 using a 1x1 convolution.

    Args:
        x: Input tensor of shape (B, H, W, C)
        name: Optional prefix for layer names

    Returns:
        Output tensor of shape (B, 2H, 2W, filters)
    """

    # Upsample spatial dimensions (bilinear)
    upsampling = layers.UpSampling2D(size=2, interpolation='bilinear')

    # Reduce channel depth using 1x1 convolution
    convolution = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding='same',
    )

    return Sequential([upsampling, convolution])

def DepthwiseSeparableConv2D(out_channels, kernel_size=3, stride=1, activation="relu", padding="same", use_bias=False):
    """
    Returns a depthwise separable convolution block: DepthwiseConv2D + Pointwise Conv2D.
    """
    depthwise_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias)
    pointwise_conv = layers.Conv2D(out_channels, kernel_size=1, strides=1, activation=activation, padding=padding, use_bias=use_bias)
    return Sequential([depthwise_conv, pointwise_conv])

# --- SSM Block (Diagonal, Efficient) ---
class SSMBlock(layers.Layer):
    """
    Implements a diagonal State Space Model (SSM) block with a scan along the width dimension.
    """
    def __init__(self, channels, ssm_state_dim=16):
        super().__init__()
        self.channels = channels
        self.ssm_state_dim = ssm_state_dim
        # Diagonal SSM parameters for each channel
        self.A = self.add_weight("A", shape=(channels, ssm_state_dim), initializer="glorot_uniform", trainable=True)
        self.B = self.add_weight("B", shape=(channels, ssm_state_dim), initializer="glorot_uniform", trainable=True)
        self.C = self.add_weight("C", shape=(channels, ssm_state_dim), initializer="glorot_uniform", trainable=True)
        self.D = self.add_weight("D", shape=(channels,), initializer="zeros", trainable=True)

    def call(self, inputs):   
        """
        inputs: Tensor of shape [batch_size, height, width, num_channels]
        Returns: Tensor of shape [batch_size, height, width, num_channels]
        """
        # inputs: [B, H, W, C]
        B, H, W, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], inputs.shape[-1]
        # Flatten spatial dims for scan
        u = tf.reshape(inputs, [B, H, W, C])
        # We'll scan along width (axis=2)
        def scan_fn(prev_tuple, u_t):
            """
            prev_tuple: (prev_state, prev_output)
                prev_state: [batch_size, height, num_channels, ssm_state_dim]
                prev_output: [batch_size, height, num_channels] (unused)
            current_input: [batch_size, height, num_channels]
            """
            prev_state, _ = prev_tuple
            # prev_state: [B, H, C, ssm_state_dim]
            # u_t: [B, H, C]
            x = prev_state * tf.reshape(self.A, [1, 1, C, self.ssm_state_dim]) + \
                tf.expand_dims(u_t, -1) * tf.reshape(self.B, [1, 1, C, self.ssm_state_dim])
            y = x * tf.reshape(self.C, [1, 1, C, self.ssm_state_dim]) + \
                tf.expand_dims(u_t, -1) * tf.reshape(self.D, [1, 1, C, 1])
            y_out = tf.reduce_sum(y, axis=-1)  # [B, H, C]
            return x, y_out

        # Transpose to [W, B, H, C]
        u_t = tf.transpose(u, [2, 0, 1, 3])
        init_state = tf.zeros([B, H, C, self.ssm_state_dim], dtype=u.dtype)
        scan_result = tf.scan(scan_fn, u_t, initializer=(init_state, tf.zeros([B, H, C], dtype=u.dtype)))
        y = scan_result[1]  # [W, B, H, C]
        y = tf.transpose(y, [1, 2, 0, 3])  # [B, H, W, C]
        return y

# --- VSS Module ---
class VSSModule(layers.Layer):
    """
    Vision State Space (VSS) module as in the diagram.
    """
    def __init__(self, num_channels, ssm_state_dim=16, expand_ratio=2, dw_kernel=3):
        super().__init__()
        self.num_channels = num_channels
        self.expand_ratio = expand_ratio
        self.ssm_state_dim = ssm_state_dim

        self.linear1 = layers.Dense(expand_ratio * num_channels)
        self.dwconv = layers.DepthwiseConv2D(kernel_size=dw_kernel, padding='same')
        self.silu1 = layers.Activation('swish')
        self.linear2 = layers.Dense(expand_ratio * num_channels)
        self.silu2 = layers.Activation('swish')
        self.ssm = SSMBlock(expand_ratio * num_channels, ssm_state_dim=ssm_state_dim)
        self.norm = layers.LayerNormalization(axis=-1)
        self.linear3 = layers.Dense(num_channels)

    def call(self, inputs):
        """
        inputs: Tensor of shape [batch_size, height, width, num_channels]
        """
        # Left chain
        x = self.linear1(inputs)
        x = self.silu1(x)
        
        # Right chain
        y = self.linear2(inputs)
        y = self.dwconv(y)
        y = self.silu2(y)
        y = self.ssm(y)
        y = self.norm(y)
        
        x = tf.multiply(x, y)
        x = self.linear3(x)
        return x

# --- RVM Layer ---
class RVMLayer(layers.Layer):
    """
    RVM Layer: LayerNorm -> VSS -> Add & Scale -> LayerNorm -> Projection
    """
    def __init__(self, num_channels, ssm_state_dim=16, expand_ratio=2, dw_kernel=3, keep_depth=False):
        super().__init__()
        self.norm1 = layers.LayerNormalization(axis=-1)
        self.vss = VSSModule(num_channels, ssm_state_dim, expand_ratio, dw_kernel)
        self.scale = self.add_weight(shape=(1,), initializer="ones", trainable=True)
        self.norm2 = layers.LayerNormalization(axis=-1)
        self.proj = layers.Dense(num_channels if keep_depth else num_channels * 2)

    def call(self, inputs):
        shortcut = inputs
        x = self.norm1(inputs)
        x = self.vss(x)
        
        scaled_shortcut = self.scale * shortcut
        x = x + scaled_shortcut
        
        x = self.norm2(x)
        x = self.proj(x)
        return x

# --- Encoder Block ---
class EncoderBlock(layers.Layer):
    """
    Encoder block: multiple RVM layers followed by MaxPooling2D.
    """
    def __init__(self, num_channels, num_rvm_layers=2, ssm_state_dim=16, expand_ratio=2, dw_kernel=3):
        super().__init__()
        self.rvm_layers = []
        for _ in range(num_rvm_layers):
            rvm_layer = RVMLayer(num_channels, ssm_state_dim, expand_ratio, dw_kernel)
            self.rvm_layers.append(rvm_layer)
            num_channels *= 2
            
        self.pool = layers.MaxPooling2D(pool_size=2)

    def call(self, inputs):
        x = inputs
        for rvm in self.rvm_layers:
            x = rvm(x)
        x = self.pool(x)
        return x

# --- Decoder Block ---
class DecoderBlock(layers.Layer):
    """
    Decoder block: upsampling, skip connection, ReLU, DWConv, scale, RVM layers.
    """
    def __init__(self, num_channels, next_num_channels, dw_kernel=3):
        super().__init__()
        self.interpolation = Interpolation(next_num_channels)
        self.relu = layers.Activation('relu')
        self.dwconv = DepthwiseSeparableConv2D(num_channels, kernel_size=dw_kernel, padding='same')
        self.scale = self.add_weight(shape=(1,), initializer="ones", trainable=True)

    def call(self, inputs, skip_connection):
        # 1. Add skip connection
        x = inputs + skip_connection
        # 2. Scale (learnable)
        scaled_x = self.scale * x
        # 3. DWConv
        x = self.dwconv(x)
        # 4. Add
        x = x + scaled_x
        # 5. ReLU
        x = self.relu(x)
        # 6. Interpolation (upsample)
        x = self.interpolation(x)
        return x

class LightMUNet(ParametersLoaderModel):
    NAME = "LightMUNet"
    NEEDS_BUILDING = True
    
    @classmethod
    def get_parameters_values(cls) -> list[Parameters]:
        return cls.generate_parameters_list(ParametersRegistry.LIGHTMUNET)
    
    @classmethod
    def show_model_summary(cls):
        model = cls()
        model.build(input_shape=(None, INPUT_HEIGHT, INPUT_WIDTH, INP_CHANNELS))
        model.summary()
        
    @property
    def inp_channels(self) -> int:
        return self.build_input_shape[-1] 
    
    """
    LightMUNet: U-Net style architecture with RVM layers, VSS modules, SSM, and DWConv.
    """
    def __init__(
        self,
        input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INP_CHANNELS),
        out_classes=OUT_MASKS,
        initial_filters=8,
        num_encoders=3,
        num_decoders=3,
        rvm_layers_per_block=2,
        ssm_state_dim=16,
        expand_ratio=2,
        dw_kernel=3,
    ):
        super().__init__()
        self.build_input_shape = (None, input_shape[0], input_shape[1], input_shape[2])
        self.input_layer = layers.InputLayer(input_shape=input_shape)
        self.dwconv_in = DepthwiseSeparableConv2D(initial_filters, kernel_size=initial_filters, padding="same")

        # Encoder path
        self.encoder_blocks = []
        filters = initial_filters
        for i in range(num_encoders):
            self.encoder_blocks.append(
                EncoderBlock(
                    num_channels=filters,
                    num_rvm_layers=rvm_layers_per_block,
                    ssm_state_dim=ssm_state_dim,
                    expand_ratio=expand_ratio,
                    dw_kernel=dw_kernel,
                )
            )
            filters *= 2 ** rvm_layers_per_block

        # Bottleneck
        self.bottleneck = RVMLayer(filters, ssm_state_dim, expand_ratio, dw_kernel, keep_depth=True)

        # Decoder path
        self.decoder_blocks = []
        for i in range(num_decoders):
            next_filters = filters // (2 ** rvm_layers_per_block)
            self.decoder_blocks.append(
                DecoderBlock(
                    num_channels=filters,
                    next_num_channels=next_filters,
                    dw_kernel=dw_kernel,
                )
            )
            filters = next_filters
            
            
        self.dwconv_out = DepthwiseSeparableConv2D(out_classes, kernel_size=dw_kernel, activation="sigmoid", padding='same')

    def call(self, inputs):
        """
        Forward pass for LightMUNet.
        """
        x = self.input_layer(inputs)
        
        # Initial convolution
        x = self.dwconv_in(x)
        
        skip_connections = []
        
        # Call encoders
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            
        # Call bottleneck
        x = self.bottleneck(x)
        
        # Call decoders
        for decoder, skip in zip(self.decoder_blocks, reversed(skip_connections)):  
            x = decoder(x, skip)
            
        # Final convolution
        x = self.dwconv_out(x)
        
        return x