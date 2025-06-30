import tensorflow as tf
from tensorflow.keras import layers, Sequential # type: ignore

from config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS, ParametersRegistry
from dataclass import Parameters
from utils.misc import ParametersLoaderModel # type: ignore

def get_depthwise_separable_conv(out_channels, kernel_size=3, stride=1, use_bias=False):
    """
    Returns a depthwise separable convolution block: DepthwiseConv2D + Pointwise Conv2D.
    """
    depthwise_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=use_bias)
    pointwise_conv = layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='same', use_bias=use_bias)
    return Sequential([depthwise_conv, pointwise_conv])

class SpatialMambaBlock(layers.Layer):
    """
    Implements a spatial Mamba block with gating and SSM scan as described in the Mamba U-Net paper.
    """
    def __init__(self, channels, ssm_state_dim=16, expand_ratio=2, conv_kernel_size=4):
        super().__init__()
        self.channels = channels
        self.expand_ratio = expand_ratio
        self.ssm_state_dim = ssm_state_dim
        self.conv_kernel_size = conv_kernel_size

        # LayerNorm before projection
        self.norm = layers.LayerNormalization(axis=-1)
        # Linear projection to higher dimension
        self.expand_proj = layers.Dense(self.expand_ratio * channels, use_bias=False)
        # Gating mechanism
        self.gate_proj = layers.Dense(self.expand_ratio * channels, use_bias=False)
        # SSM parameters (A, B, C, D) for each channel
        self.A = self.add_weight("A", shape=(self.expand_ratio * channels, ssm_state_dim), initializer="glorot_uniform", trainable=True)
        self.B = self.add_weight("B", shape=(self.expand_ratio * channels, ssm_state_dim), initializer="glorot_uniform", trainable=True)
        self.C = self.add_weight("C", shape=(self.expand_ratio * channels, ssm_state_dim), initializer="glorot_uniform", trainable=True)
        self.D = self.add_weight("D", shape=(self.expand_ratio * channels,), initializer="zeros", trainable=True)
        # Output projection
        self.output_proj = layers.Dense(channels, use_bias=False)

    def ssm_scan(self, u):
        """
        Performs the SSM scan along the spatial dimension (width).
        u: [B, H, W, C]
        Returns: [B, H, W, C]
        """
        # We'll scan along the width (axis=2)
        def scan_fn(prev_state, inputs):
            prev_state, _ = prev_state  # Unpack the tuple
            # prev_state: [B, H, C, ssm_state_dim]
            # inputs: [B, H, C]
            #print(f"{prev_state.shape} {self.A.shape} {self.B.shape}")
            assert prev_state.shape[-2] == self.expand_ratio * self.channels
            assert inputs.shape[-1] == self.expand_ratio * self.channels  
            # SSM: x_t = A * x_{t-1} + B * u_t
            # y_t = C * x_t + D * u_t
            x = prev_state * tf.reshape(self.A, [1, 1, -1, self.ssm_state_dim]) + \
                tf.expand_dims(inputs, -1) * tf.reshape(self.B, [1, 1, -1, self.ssm_state_dim])
            y = x * tf.reshape(self.C, [1, 1, -1, self.ssm_state_dim]) + \
                tf.expand_dims(inputs, -1) * tf.reshape(self.D, [1, 1, -1, 1])
            # For output, sum over state dim
            y_out = tf.reduce_sum(y, axis=-1)  # [B, H, C]
            return x, y_out

        batch_size = tf.shape(u)[0]
        height = tf.shape(u)[1]
        width = tf.shape(u)[2]
        channels = u.shape[-1]
        # Transpose to [W, B, H, C]
        u_t = tf.transpose(u, [2, 0, 1, 3])
        # Initial state: zeros [B, H, C, ssm_state_dim]
        init_state = tf.zeros([batch_size, height, channels, self.ssm_state_dim], dtype=u.dtype)
        # Run scan
        _, y = tf.scan(scan_fn, u_t, initializer=(init_state, tf.zeros([batch_size, height, channels], dtype=u.dtype)))
        # y: [W, B, H, C]
        y = tf.transpose(y, [1, 2, 0, 3])  # [B, H, W, C]
        return y

    def call(self, inputs):
        # inputs: [B, H, W, C]
        x = self.norm(inputs)
        x_proj = self.expand_proj(x)  # [B, H, W, expand*C]
        gate = tf.sigmoid(self.gate_proj(x))  # [B, H, W, expand*C]
        # SSM scan along width
        ssm_out = self.ssm_scan(x_proj)  # [B, H, W, expand*C]
        # Gated output
        gated_out = gate * ssm_out
        # Project back to original channels
        out = self.output_proj(gated_out)
        return out

class MambaResidualLayer(layers.Layer):
    """
    Residual block with spatial Mamba SSM logic.
    """
    def __init__(self, channels, expand_ratio=2, ssm_state_dim=16, conv_kernel_size=4):
        super().__init__()
        self.norm1 = layers.LayerNormalization(axis=-1)
        self.norm2 = layers.LayerNormalization(axis=-1)
        self.activation = layers.Activation('gelu')
        self.mamba1 = SpatialMambaBlock(channels, ssm_state_dim=ssm_state_dim, expand_ratio=expand_ratio, conv_kernel_size=conv_kernel_size)
        self.mamba2 = SpatialMambaBlock(channels, ssm_state_dim=ssm_state_dim, expand_ratio=expand_ratio, conv_kernel_size=conv_kernel_size)

    def call(self, inputs):
        shortcut = inputs
        x = self.norm1(inputs)
        x = self.activation(x)
        x = self.mamba1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.mamba2(x)
        return x + shortcut

class ResidualUpBlock(layers.Layer):
    """
    Residual upsampling block using depthwise separable convolution.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.norm1 = layers.LayerNormalization(axis=-1)
        self.norm2 = layers.LayerNormalization(axis=-1)
        self.activation = layers.Activation('relu')
        self.conv = get_depthwise_separable_conv(channels, kernel_size=kernel_size)
        self.skip_scale = self.add_weight(shape=(1,), initializer="ones", trainable=True)

    def call(self, inputs):
        shortcut = inputs
        x = self.norm1(inputs)
        x = self.activation(x)
        x = self.conv(x) + self.skip_scale * shortcut
        x = self.norm2(x)
        x = self.activation(x)
        return x

class LightMUNet(ParametersLoaderModel):
    NAME = "LightMUNet"
    
    @classmethod
    def get_parameters_values(cls) -> list[Parameters]:
        return cls.generate_parameters_list(ParametersRegistry.LIGHTMUNET)
    
    @classmethod
    def show_model_summary(cls):
        model = cls()
        model.build(input_shape=(None, INPUT_HEIGHT, INPUT_WIDTH, INP_CHANNELS))
        model.summary()
    
    """
    LightMUNet: U-Net style architecture with spatial Mamba SSM blocks and depthwise separable convolutions.
    """
    def __init__(
        self,
        initial_filters=16,
        input_channels=INP_CHANNELS,
        output_channels=OUT_MASKS,
        dropout_rate=None,
        use_final_conv=True,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    ):
        super().__init__()
        self.initial_filters = initial_filters
        self.input_channels = input_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_rate = dropout_rate
        self.use_final_conv = use_final_conv

        # Initial depthwise separable convolution
        self.initial_conv = get_depthwise_separable_conv(initial_filters)
        # Downsampling path
        self.down_layers = self._make_down_layers()
        # Upsampling path
        self.up_layers, self.up_samples = self._make_up_layers()
        # Final output convolution
        self.final_conv = self._make_final_conv(output_channels)
        if dropout_rate is not None:
            self.dropout = layers.Dropout(dropout_rate)

    def _make_down_layers(self):
        """
        Constructs the downsampling path with Mamba residual blocks.
        """
        down_layers = []
        filters = self.initial_filters
        for i, num_blocks in enumerate(self.blocks_down):
            layer_channels = filters * 2 ** i
            if i > 0:
                # Downsample with stride-2 max pooling
                downsample = Sequential([
                    layers.Conv2D(layer_channels, kernel_size=1, strides=2, padding='same'),
                    layers.LayerNormalization(axis=-1),
                    layers.Activation('relu')
                ])
            else:
                downsample = layers.Lambda(lambda x: x)
            block_layers = [MambaResidualLayer(layer_channels) for _ in range(num_blocks)]
            down_layers.append(Sequential([downsample] + block_layers))
        return down_layers

    def _make_up_layers(self):
        """
        Constructs the upsampling path with residual up blocks and upsampling layers.
        """
        up_layers = []
        up_samples = []
        num_ups = len(self.blocks_up)
        filters = self.initial_filters
        for i in range(num_ups):
            sample_channels = filters * 2 ** (num_ups - i)
            block_layers = [ResidualUpBlock(sample_channels // 2) for _ in range(self.blocks_up[i])]
            up_layers.append(Sequential(block_layers))
            up_samples.append(
                Sequential([
                    layers.Conv2D(sample_channels // 2, kernel_size=1, padding='same'),
                    layers.UpSampling2D(size=2, interpolation='nearest')
                ])
            )
        return up_layers, up_samples

    def _make_final_conv(self, output_channels):
        """
        Final normalization, activation, and 1x1 convolution to produce output.
        """
        return Sequential([
            layers.LayerNormalization(axis=-1),
            layers.Activation('relu'),
            layers.Conv2D(output_channels, kernel_size=1, padding='same')
        ])

    def encode(self, inputs):
        """
        Forward pass through the encoder (downsampling path).
        """
        x = self.initial_conv(inputs)
        if self.dropout_rate is not None:
            x = self.dropout(x)
        skip_connections = []
        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)
        return x, skip_connections

    def decode(self, x, skip_connections):
        """
        Forward pass through the decoder (upsampling path).
        """
        for i, (upsample, up_block) in enumerate(zip(self.up_samples, self.up_layers)):
            x = upsample(x) + skip_connections[i + 1]
            x = up_block(x)
        if self.use_final_conv:
            x = self.final_conv(x)
        return x

    def call(self, inputs):
        """
        Full forward pass.
        """
        x, skip_connections = self.encode(inputs)
        skip_connections = skip_connections[::-1]  # Reverse for skip connections
        x = self.decode(x, skip_connections)
        return x