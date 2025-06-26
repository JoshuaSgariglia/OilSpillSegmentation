import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers # type: ignore
from keras.models import Model
from keras.layers import Input, Layer, Dropout
from config import INP_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, OUT_MASKS, ParametersRegistry
from dataclass import Parameters
from utils.misc import ParametersLoaderModel # type: ignore

class SS2D(Layer):
    def __init__(self, d_model, dt_rank="auto", dt_scale=1.0, dt_init="random",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                 dropout_rate=0.0, expand=2, **kwargs):
        """
        SS2D Layer using a custom Dense projection with Softplus-inverted bias initialization.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dt_scale = dt_scale
        self.dt_init = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dropout_rate = dropout_rate
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.dt_proj = None
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0. else None

    def build(self, input_shape):
        """
        Build the projection layer with custom weight and bias initialization.
        """
        # Weight initialization
        std = (self.dt_rank ** -0.5) * self.dt_scale
        if self.dt_init == "constant":
            kernel_init = initializers.Constant(std)
        elif self.dt_init == "random":
            limit = std
            kernel_init = initializers.RandomUniform(minval=-limit, maxval=limit)
        else:
            raise NotImplementedError(f"dt_init='{self.dt_init}' not supported")

        # Custom bias initializer using inverse softplus
        def bias_init(shape, dtype=None):
            rand_uniform = np.random.rand(self.d_inner)
            log_range = np.log(self.dt_max) - np.log(self.dt_min)
            dt = np.exp(rand_uniform * log_range + np.log(self.dt_min))
            dt = np.clip(dt, self.dt_init_floor, None)

            # Inverse of softplus
            inv_dt = dt + np.log(-np.expm1(-dt))
            return tf.convert_to_tensor(inv_dt, dtype=dtype)

        # Define the Dense layer manually
        self.dt_proj = layers.Dense(self.d_inner,
                                    use_bias=True,
                                    kernel_initializer=kernel_init,
                                    bias_initializer=bias_init)

    def call(self, x, training=False):
        """
        Forward pass through the SS2D dt projection.
        Optionally applies dropout.
        """
        x = self.dt_proj(x)
        if self.dropout:
            x = self.dropout(x, training=training)
        return x
    
    import tensorflow as tf

class PatchEmbed2D(Layer):
    """
    Converts an image to patch embeddings for Vision Transformers.

    Args:
        patch_size (int or tuple): Size of each patch. Default is 4.
        in_chans (int): Number of input channels. Default is 3.
        embed_dim (int): Dimension of output embeddings. Default is 96.
        norm_layer (tf.keras.layers.Layer, optional): Optional normalization layer (e.g. LayerNormalization).
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super(PatchEmbed2D, self).__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',  # No overlap between patches
            input_shape=(None, None, in_chans)
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def call(self, x):
        """
        x: Tensor of shape [B, H, W, C] (standard TensorFlow format)
        Returns:
            Tensor of shape [B, H//patch, W//patch, embed_dim]
        """
        x = self.proj(x)  # [B, H//P, W//P, embed_dim]
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class PatchMerging2D(Layer):
    """
    Patch Merging Layer for hierarchical Vision Transformers.
    
    Args:
        dim (int): Number of input channels.
        norm_layer (tf.keras.layers.Layer): Normalization layer (e.g., LayerNormalization).
    """
    def __init__(self, dim, norm_layer=None):
        super(PatchMerging2D, self).__init__()
        self.dim = dim
        self.norm = norm_layer if norm_layer is not None else layers.LayerNormalization(epsilon=1e-6)
        self.reduction = layers.Dense(2 * dim, use_bias=False)  # Linear projection to reduce dimensionality

    def call(self, x):
        """
        x: Tensor of shape [B, H, W, C]
        Returns:
            Tensor of shape [B, H//2, W//2, 2*C]
        """
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Pad if height or width is odd
        pad_input = (H % 2 != 0) or (W % 2 != 0)
        if pad_input:
            x = tf.pad(x, [[0, 0], [0, H % 2], [0, W % 2], [0, 0]])

        # Extract patches from 2x2 blocks
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right

        # Concatenate along channel dimension: [B, H//2, W//2, 4*C]
        x = tf.concat([x0, x1, x2, x3], axis=-1)

        # Normalize and project to new dimension
        x = self.norm(x)
        x = self.reduction(x)  # Now shape is [B, H//2, W//2, 2*C]

        return x

class DropPath(Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        """
        DropPath (Stochastic Depth): Randomly drops residual paths during training.

        Args:
            drop_prob (float): Probability of dropping paths.
        """
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if (not training) or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Compute shape: (batch_size, 1, 1, 1) to broadcast across all spatial dims
        shape = tf.shape(x)
        broadcast_shape = [shape[0]] + [1] * (len(shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(broadcast_shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        x = tf.math.divide(x, keep_prob) * binary_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config    
    
class VSSBlock(Layer):
    def __init__(self, hidden_dim, drop_path=0.0, attn_drop_rate=0.0,
                 norm_layer='layer_norm', d_state=16, **kwargs):
        """
        VSSBlock: One block in a Vision Structured State Space model.

        Args:
            hidden_dim (int): Input and output feature dimension.
            drop_path (float): Stochastic depth rate.
            attn_drop_rate (float): Dropout rate inside SS2D.
            norm_layer (str): Type of normalization (e.g., 'layer_norm').
            d_state (int): State dimension inside SS2D.
        """
        super().__init__(**kwargs)

        # Normalization layer
        self.norm = layers.LayerNormalization(epsilon=1e-6) if norm_layer == 'layer_norm' else tf.identity

        # State-space/self-attention module
        self.self_attention = SS2D(d_model=hidden_dim, dropout_rate=attn_drop_rate, **kwargs)

        # DropPath: for stochastic depth regularization
        self.drop_path = DropPath(drop_path)

    def call(self, x, training=False):
        """
        Forward pass with residual connection.
        """
        shortcut = x
        x = self.norm(x)
        x = self.self_attention(x, training=training)
        x = self.drop_path(x, training=training)
        
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv2D(x.shape[-1], kernel_size=1, padding='same')(shortcut)
        
        return shortcut + x    
    
class VSSLayer(Layer):
    def __init__(self, dim, depth, attn_drop=0.0, drop_path=0.0,
                 norm_layer='layer_norm', downsample=None,
                 d_state=16, **kwargs):
        """
        VSSLayer: A sequence of VSSBlocks (like a Transformer stage).

        Args:
            dim (int): Feature/channel dimension.
            depth (int): Number of VSSBlocks.
            attn_drop (float): Dropout rate for SS2D attention.
            drop_path (float or list): DropPath rate or list per block.
            norm_layer (str): Normalization type.
            downsample (Layer or None): Optional downsampling after blocks.
            d_state (int): SS2D state size.
        """
        super().__init__(**kwargs)

        self.blocks = []

        # Handle variable drop_path per block
        if isinstance(drop_path, (list, tuple)):
            drop_path_list = drop_path
        else:
            drop_path_list = [drop_path] * depth

        for i in range(depth):
            blk = VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_list[i],
                attn_drop_rate=attn_drop,
                norm_layer=norm_layer,
                d_state=d_state
            )
            self.blocks.append(blk)

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def call(self, x, training=False):
        """
        Forward through all blocks, then apply downsampling if specified.
        """
        for blk in self.blocks:
            x = blk(x, training=training)

        if self.downsample is not None:
            x = self.downsample(x)

        return x    
    
class VSSM(Layer):
    def __init__(
        self,
        norm_layer=None,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # Patch embedding
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)


        self.ape = False  # Absolute position embedding

        self.pos_drop = Dropout(drop_rate)

        # Compute stochastic depth decay for each block
        dpr = tf.linspace(0.0, drop_path_rate, sum(depths)).numpy().tolist()

        # Encoder: stack of VSSLayer
        self.layers = []
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state if d_state is not None else math.ceil(dims[0] / 6),
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=layers.LayerNormalization,
                downsample=None if i_layer == self.num_layers - 1 else kwargs.get("PatchMerging2D"),
            )
            self.layers.append(layer)

    def call(self, x, training=False):
        skip_list = []

        # Apply patch embedding
        x = self.patch_embed(x)

        # Optionally add absolute position embedding
        if self.ape:
            x += self.absolute_pos_embed

        x = self.pos_drop(x, training=training)

        # Apply VSSLayers
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x, training=training)

        # Return skip connections for decoder use in segmentation
        return skip_list[0], skip_list[1], skip_list[2], skip_list[3]
    
class ChannelAttention(Layer):
    """
    Channel Attention Module from CBAM.
    Applies both average and max pooling across spatial dimensions, then a shared MLP to compute attention weights.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)

        self.fc1 = layers.Conv2D(in_planes // ratio, kernel_size=1, use_bias=False)
        self.relu1 = layers.ReLU()
        self.fc2 = layers.Conv2D(in_planes, kernel_size=1, use_bias=False)

        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)  
  
class SpatialAttention(Layer):
    """
    Spatial Attention Module from CBAM.
    Computes attention by convolving over average and max pooled features across channels.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False
        )
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.conv(x)
        return self.sigmoid(x)  
  
class BasicConv2d(Layer):
    """
    Basic 2D Convolution Block: Conv -> BatchNorm -> ReLU
    """
    def __init__(self, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = layers.Conv2D(
            filters=out_planes,
            kernel_size=kernel_size,
            strides=stride,
            padding='same' if padding > 0 else 'valid',
            dilation_rate=dilation,
            use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
  
class SDI(Layer):
    """
    Spatial Dynamic Integration (SDI) Block.
    Combines multiple feature maps (e.g., from encoder stages) via multiplicative fusion after resizing.
    """
    def __init__(self, channel):
        super(SDI, self).__init__()
        self.convs = [layers.Conv2D(channel, kernel_size=3, padding='same') for _ in range(4)]

    def call(self, xs, anchor):
        # xs: list of feature maps from different levels [f1, f2, f3, f4]
        # anchor: reference feature map for spatial alignment
        target_size = tf.shape(anchor)[1:3]
        ans = tf.ones_like(anchor)

        for i, x in enumerate(xs):
            h, w = tf.shape(x)[1], tf.shape(x)[2]
            # Resize input to match anchor spatial size
            if h > target_size[0]:
                x = tf.image.resize(x, target_size, method='area')
            elif h < target_size[0]:
                x = tf.image.resize(x, target_size, method='bilinear')

            ans = ans * self.convs[i](x)

        return ans
    
class VMUNetV2(ParametersLoaderModel):
    NAME = "VMUNetV2"
    
    @classmethod
    def get_parameters_values(cls) -> list[Parameters]:
        return cls.generate_parameters_list(ParametersRegistry.VMUNETV2)
    
    @classmethod
    def show_model_summary(cls):
        model = cls()
        model.build(input_shape=(None, INPUT_HEIGHT, INPUT_WIDTH, INP_CHANNELS))
        model.summary()
    
    def __init__(
        self,
        input_channels=INP_CHANNELS,
        num_classes=OUT_MASKS,
        mid_channel=48,
        depths=[2, 2, 9, 2],
        drop_path_rate=0.2,
        deep_supervision=True
    ):
        super(VMUNetV2, self).__init__()

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Channel + Spatial Attention
        self.ca_1 = ChannelAttention(2 * mid_channel)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(4 * mid_channel)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(8 * mid_channel)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(16 * mid_channel)
        self.sa_4 = SpatialAttention()

        # Transition Layers
        self.Translayer_1 = BasicConv2d(2 * mid_channel, mid_channel, 1)
        self.Translayer_2 = BasicConv2d(4 * mid_channel, mid_channel, 1)
        self.Translayer_3 = BasicConv2d(8 * mid_channel, mid_channel, 1)
        self.Translayer_4 = BasicConv2d(16 * mid_channel, mid_channel, 1)

        # SDI Modules
        self.sdi_1 = SDI(mid_channel)
        self.sdi_2 = SDI(mid_channel)
        self.sdi_3 = SDI(mid_channel)
        self.sdi_4 = SDI(mid_channel)

        # Segmentation heads (1x1 convs)
        self.seg_outs = [layers.Conv2D(num_classes, kernel_size=1) for _ in range(4)]

        # Upsample layers
        self.deconv2 = layers.Conv2DTranspose(mid_channel, 4, strides=2, padding='same', use_bias=False)
        self.deconv3 = layers.Conv2DTranspose(mid_channel, 4, strides=2, padding='same', use_bias=False)
        self.deconv4 = layers.Conv2DTranspose(mid_channel, 4, strides=2, padding='same', use_bias=False)
        self.deconv6 = layers.Conv2DTranspose(1, 3, strides=2, padding='same', output_padding=1)

        # Base encoder-decoder (defined externally)
        self.vmunet = VSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            drop_path_rate=drop_path_rate
        )

    def call(self, x, training=False):
        seg_outs = []

        if x.shape[3] == 1:
            x = tf.repeat(x, 3, axis=3)  # Grayscale to RGB

        f1, f2, f3, f4 = self.vmunet(x)  # Outputs: [B, H, W, C]

        # Apply attention + transition
        f1 = self.Translayer_1(self.sa_1(self.ca_1(f1) * f1) * f1)
        f2 = self.Translayer_2(self.sa_2(self.ca_2(f2) * f2) * f2)
        f3 = self.Translayer_3(self.sa_3(self.ca_3(f3) * f3) * f3)
        f4 = self.Translayer_4(self.sa_4(self.ca_4(f4) * f4) * f4)
        
        # Permute to [B, C, H, W] equivalent (for attention to work)
        #f1 = tf.transpose(f1, [0, 3, 2, 1])
        #f2 = tf.transpose(f2, [0, 3, 2, 1])
        #f3 = tf.transpose(f3, [0, 3, 1, 2])
        #f4 = tf.transpose(f4, [0, 3, 1, 2])

        # SDI integration
        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)

        # Segmentation outputs with upsampling chain
        y = tf.transpose(f41, [0, 2, 3, 1])
        seg_outs.append(self.seg_outs[0](y))
        
        y = self.deconv2(tf.transpose(f41, [0, 2, 3, 1])) + tf.transpose(f31, [0, 2, 3, 1])
        seg_outs.append(self.seg_outs[1](y))

        y = self.deconv3(y) + tf.transpose(f21, [0, 2, 3, 1])
        seg_outs.append(self.seg_outs[2](y))

        y = self.deconv4(y) + tf.transpose(f11, [0, 2, 3, 1])
        seg_outs.append(self.seg_outs[3](y))

        # Final interpolation to input size (assume 256x256 for now)
        seg_outs = [tf.image.resize(o, size=(x.shape[1], x.shape[2]), method='bilinear') for o in seg_outs]

        if self.deep_supervision:
            out_0 = seg_outs[-1]  # Largest resolution
            out_1 = self.deconv6(seg_outs[-2])
            return tf.keras.activations.sigmoid(out_0 + out_1)
        else:
            out = seg_outs[-1]
            return tf.keras.activations.sigmoid(out) if self.num_classes == 1 else out    
    
    
    
    
    