import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()  

from tensorflow.keras.layers import Dense, LayerNormalization, Embedding, Flatten
from tensorflow.keras.models import Model

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f'divide error')
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    
    def BuildAttention(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    
    def BuildTransformerBlock(self, inputs, training):
        attn_output = self.att.BuildAttention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class VisionTransformer(Model):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim):
        super(VisionTransformer, self).__init__()
        self.num_patches = num_patches
        embed_dim=embed_dim-embed_dim%num_heads
        # if embed_dim%num_heads!=0:
        #     if embed_dim-embed_dim%num_heads>0:
        #         embed_dim=embed_dim-embed_dim%num_heads
        #     else:
        #         embed_dim=embed_dim+num_heads-embed_dim%num_heads
            
        
            
        self.d_embedding = Dense(embed_dim)
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches, embed_dim))
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        

    
    def BuildVisionTransformer(self, x, training):
        x = self.d_embedding(x)
        x += self.pos_emb
        x = self.transformer_block.BuildTransformerBlock(x, training=training)
        return x

# class PatchExpansion(tf.keras.layers.Layer):
#     def __init__(self, dim):
#         super(PatchExpansion, self).__init__()
#         self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
#         self.expand = tf.keras.layers.Dense(2 * dim, use_bias=False)

#     def BuildPatchExpansion(self, x):
#         x = self.expand(x)
#         B, H, W, C = x.get_shape().as_list()
#         x = tf.reshape(x, [B, H, W, 2, 2, C // 4])
#         x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
#         x = tf.reshape(x, [B, H * 2, W * 2, C // 4])
#         x = self.norm(x)
#         return x

# class PatchMerging(tf.keras.layers.Layer):
#     def __init__(self, dim):
#         super(PatchMerging, self).__init__()
#         self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
#         self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False)

#     def BuildPatchMerging(self, x):
#         B, H, W, C = x.get_shape().as_list()
#         x = tf.reshape(x, [B, H // 2, 2, W // 2, 2, C])
#         x = tf.transpose(x, [0, 1, 3, 4, 2, 5])
#         x = tf.reshape(x, [B, -1, 4 * C])
#         x = self.norm(x)
#         x = self.reduction(x)
#         x = tf.reshape(x, [B, H//2, W//2, C])
#         return x


    
class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=tf.keras.layers.LayerNormalization, dim_scale=2):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale=dim_scale
        self.reduction = tf.keras.layers.Dense(dim_scale * dim, use_bias=False)
        self.norm = norm_layer(axis=-1)

    def BuildPatchMerging(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, [B, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
        x = tf.reshape(x, [B, -1, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpansion(tf.keras.layers.Layer):
    # Example usage:
    # Assuming 'input_resolution' is (H, W), 'dim' is the dimension of the input features,
    # and 'dim_scale' is the scale factor for dimensionality expansion.
    # Here's how you can use the PatchExpand module:

    # Define input tensor 'x' with shape (B, H * W, C)
    # where B is the batch size, H and W are the height and width of the grid, and C is the input feature dimension.

    # patch_expand = PatchExpand(input_resolution=(H, W), dim=C, dim_scale=2)
    # output = patch_expand(x)
    # 'output' will have the shape (B, 2 * H * W, C // 4), which is the expanded patch representation.

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=tf.keras.layers.LayerNormalization):
        super(PatchExpansion, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = tf.keras.layers.Dense(2 * dim, use_bias=False) if dim_scale == 2 else tf.keras.layers.Lambda(lambda x: x)
        self.norm = norm_layer(axis=-1)

    def BuildPatchExpansion(self, x):
        """
        x: B, H*W, c
        """
        H, W = self.input_resolution

        x = self.expand(x)  # [B,H*W,2c]
        B, L, C = int(x.shape[0]), int(x.shape[1]),int(x.shape[2])
        assert L == H * W, "input feature has wrong size"

        x = tf.reshape(x, [B, H, W, C])
        x = tf.reshape(x, [B, H, W, -1, C // 4])
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        x = tf.reshape(x, [B, -1, W * 2, C // 4])
        x = tf.reshape(x, [B, -1, C // 4])
        x = self.norm(x)
        return x


    
def VitImplementation(x, scope, numVit, embedDim, numHeads, ffDim, training):    
    thisInput = x
    for ii in range(numVit):
        with tf.variable_scope("Vit%02d"%(ii+1)):
            vit = VisionTransformer(int(x.shape[1]), embedDim, numHeads, ffDim)
            vitFeature = vit.BuildVisionTransformer(x=thisInput, training=training)
            thisInput=vitFeature    
    return vitFeature

def PatchMergingImplementation(x, input_resolution, dim, dim_scale=2):
    merger = PatchMerging(input_resolution=input_resolution, dim=dim, dim_scale=dim_scale)
    result = merger.BuildPatchMerging(x=x)
    return result

def PatchExpandingImplementation(x, input_resolution, dim):
    expander = PatchExpansion(input_resolution=input_resolution, dim=dim)
    result = expander.BuildPatchExpansion(x=x)
    return result