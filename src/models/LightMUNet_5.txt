import tensorflow as tf
from tensorflow.keras import layers, Sequential # type: ignore

from config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS, ParametersRegistry
from dataclass import Parameters
from utils.misc import ParametersLoaderModel # type: ignore

def Interpolation(filters, name):
    """
    Upscales H and W by 2x using bilinear interpolation,
    and reduces the channel dimension by 4 using a 1x1 convolution.

    Args:
        x: Input tensor of shape (B, H, W, C)
        name: Optional prefix for layer names

    Returns:
        Output tensor of shape (B, 2H, 2W, filters)
    """
    if name is None:
        name = f"interpolation_{filters}"

    # Upsample spatial dimensions (bilinear)
    upsampling = layers.UpSampling2D(size=2, interpolation='bilinear')

    # Reduce channel depth using 1x1 convolution
    convolution = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding='same',
    )

    return Sequential([upsampling, convolution], name=name)

def DepthwiseSeparableConv2D(out_channels, name, kernel_size=3, stride=1, activation="relu", padding="same", use_bias=False):
    """
    Returns a depthwise separable convolution block: DepthwiseConv2D + Pointwise Conv2D.
    """
    if name is None:
        name = f"depth_sep_conv_{out_channels}"
    
    depthwise_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias)
    pointwise_conv = layers.Conv2D(out_channels, kernel_size=1, strides=1, activation=activation, padding=padding, use_bias=use_bias)
    return Sequential([depthwise_conv, pointwise_conv], name=name)

# --- SSM Block ---
class SSMBlock(layers.Layer):
    """
    Efficient diagonal SSM block using tf.scan for fast sequence modeling.
    Implements a Mamba-style block using learned per-step diagonal state-space updates.
    """

    def __init__(self, channels, ssm_state_dim):
        """
        Args:
            channels (int): Number of input/output feature channels (C).
            ssm_state_dim (int): Dimension of the SSM hidden state per channel (S).
        """
        super().__init__()
        self.channels = channels
        self.state_dim = ssm_state_dim

        # Input projection to produce:
        # - value, gate, delta (C each)
        # - A_mat, B_mat, C_mat (C * S each)
        total_proj = 3 * channels + 3 * channels * ssm_state_dim
        self.input_proj = layers.Dense(total_proj, use_bias=True)

        # Output projection (mix channels after SSM processing)
        self.output_proj = layers.Dense(channels, use_bias=True)

        # Layer normalization (as in Mamba)
        self.norm = layers.LayerNormalization(axis=-1)

    def call(self, inputs):
        """
        Args:
            inputs: Tensor of shape [B, H, W, C]
        Returns:
            Output: Tensor of shape [B, H, W, C]
        """
        B = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        C = self.channels
        S = self.state_dim
        seq_len = H * W

        # 1. Normalize input
        x = self.norm(inputs)  # [B, H, W, C]

        # 2. Flatten spatial dims into sequence: [B, seq_len, C]
        x = tf.reshape(x, [B, seq_len, C])

        # 3. Linear projection to get all SSM parameters
        proj = self.input_proj(x)  # [B, seq_len, total_proj]

        # 4. Split projection into components
        split_sizes = [C, C, C, C * S, C * S, C * S]
        value, gate, delta, A_mat, B_mat, C_mat = tf.split(proj, split_sizes, axis=-1)

        # 5. Activation for gate
        gate = tf.sigmoid(gate)

        # 6. Reshape A/B/C matrices to [B, seq_len, C, S]
        A_mat = tf.reshape(A_mat, [B, seq_len, C, S])
        B_mat = tf.reshape(B_mat, [B, seq_len, C, S])
        C_mat = tf.reshape(C_mat, [B, seq_len, C, S])

        # 7. Expand dims for value/delta: [B, seq_len, C, 1]
        value = tf.expand_dims(value, -1)
        delta = tf.expand_dims(delta, -1)

        # 8. Transpose to time-major: [seq_len, B, C, S] or [seq_len, B, C, 1]
        A_mat = tf.transpose(A_mat, [1, 0, 2, 3])
        B_mat = tf.transpose(B_mat, [1, 0, 2, 3])
        C_mat = tf.transpose(C_mat, [1, 0, 2, 3])
        value = tf.transpose(value, [1, 0, 2, 3])
        delta = tf.transpose(delta, [1, 0, 2, 3])
        gate = tf.transpose(gate, [1, 0, 2])  # [seq_len, B, C]

        # 9. Initial hidden state and output: [B, C, S] and [B, C]
        h0 = tf.zeros([B, C, S], dtype=tf.float32)
        y0 = tf.zeros([B, C], dtype=tf.float32)
        initializer = (h0, y0)

        # 10. Step function for tf.scan
        def ssm_step(carry, inputs):
            h, _ = carry
            A_t, B_t, C_t, d_t, u_t, g_t = inputs
            h_new = d_t * (A_t * h) + d_t * (B_t * u_t)  # [B, C, S]
            y_t = tf.reduce_sum(C_t * h_new, axis=-1)   # [B, C]
            y_t = g_t * y_t                              # [B, C]
            return (h_new, y_t)

        # 11. Run scan over time steps
        elems = (A_mat, B_mat, C_mat, delta, value, gate)
        h_final, y_all = tf.scan(fn=ssm_step, elems=elems, initializer=initializer)

        # 12. Transpose back to batch-major: [B, seq_len, C]
        y_all = tf.transpose(y_all, [1, 0, 2])

        # 13. Final projection
        y_all = self.output_proj(y_all)  # [B, seq_len, C]

        # 14. Reshape to spatial dimensions: [B, H, W, C]
        out = tf.reshape(y_all, [B, H, W, C])
        return out

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
        self.dwconv = DepthwiseSeparableConv2D(
            expand_ratio * num_channels, 
            kernel_size=dw_kernel, 
            padding='same',
            name=f"depth_sep_conv_{str(num_channels)}"
            )
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
        self.scale = self.add_weight(shape=(1,), name=f"weight_{str(num_channels)}", initializer="ones", trainable=True)
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
        num_channels_list: list[int] = [num_channels, num_channels * 2]
        
        for i in range(num_rvm_layers):
            setattr(self, f"rvm_block_{i}", RVMLayer(
                num_channels_list[i], ssm_state_dim, expand_ratio, dw_kernel))
        
        self._num_rvm_layers = num_rvm_layers  # Store as an int, which is safe!
            
        self.pool = layers.MaxPooling2D(pool_size=2)

    def call(self, inputs):
        x = inputs
        for i in range(self._num_rvm_layers):
            rvm = getattr(self, f"rvm_block_{i}")
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
        self.interpolation = Interpolation(next_num_channels, f"interpolation_{str(next_num_channels)}")
        self.relu = layers.Activation('relu')
        self.dwconv = DepthwiseSeparableConv2D(
            num_channels, 
            kernel_size=dw_kernel, 
            padding='same',
            name=f"depth_sep_conv_{str(num_channels)}"
            )
        self.scale = self.add_weight(shape=(1,), name=f"weight_{str(num_channels)}", initializer="ones", trainable=True)

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
        ssm_state_dim=8,
        expand_ratio=2,
        dw_kernel=3,
    ):
        super().__init__()
        self.build_input_shape = (None, input_shape[0], input_shape[1], input_shape[2])
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.input_layer = layers.InputLayer(input_shape=input_shape)
        self.dwconv_in = DepthwiseSeparableConv2D(
            initial_filters, 
            kernel_size=initial_filters, 
            padding="same",
            name="depth_sep_conv_input"
            )

        # Filters list
        filters = initial_filters
        
        filters_list: list[int] = [initial_filters]
        for i in range(num_encoders):
            filters *= 2 ** rvm_layers_per_block
            filters_list.append(filters)

        # Encoder path
        for i in range(num_encoders):
            setattr(self, f"encoder_block_{i}", EncoderBlock(
                num_channels=filters_list[i],
                num_rvm_layers=rvm_layers_per_block,
                ssm_state_dim=ssm_state_dim,
                expand_ratio=expand_ratio,
                dw_kernel=dw_kernel,
                ))

        # Bottleneck
        self.bottleneck = RVMLayer(filters, ssm_state_dim, expand_ratio, dw_kernel, keep_depth=True)
        
        filters_list = list(reversed(filters_list))

        # Decoder path
        for i in range(num_decoders):
            setattr(self, f"decoder_block_{i}", DecoderBlock(
                num_channels=filters_list[i],
                next_num_channels=filters_list[i+1],
                dw_kernel=dw_kernel,
                ))
            
        # Final convolution
        self.dwconv_out = DepthwiseSeparableConv2D(
            out_classes, 
            kernel_size=dw_kernel, 
            activation="sigmoid", 
            padding='same',
            name=f"depth_sep_conv_output"
            )

    def call(self, inputs):
        """
        Forward pass for LightMUNet.
        """
        x = self.input_layer(inputs)
        
        # Initial convolution
        x = self.dwconv_in(x)
        
        skip_connections = []
        
        # Call encoders
        for i in range(self.num_encoders):
            encoder = getattr(self, f"encoder_block_{i}")
            x = encoder(x)
            skip_connections.append(x)
            
        # Call bottleneck
        x = self.bottleneck(x)
        
        # Call decoders
        for i in range(self.num_decoders):
            decoder = getattr(self, f"decoder_block_{i}")
            x = decoder(x, skip_connections[-(i+1)])
            
        # Final convolution
        x = self.dwconv_out(x)
        
        return x