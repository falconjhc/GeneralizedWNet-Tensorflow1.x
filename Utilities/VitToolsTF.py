import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
import numpy as np

class LayerNormalization(tf.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_variable(name='gamma', shape=input_shape[-1:], initializer=tf.ones_initializer())
        self.beta = self.add_variable(name='beta', shape=input_shape[-1:], initializer=tf.zeros_initializer())
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

class MultiHeadSelfAttention(tf.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f'divide error')
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.layers.Dense(embed_dim)
        self.key_dense = tf.layers.Dense(embed_dim)
        self.value_dense = tf.layers.Dense(embed_dim)
        self.combine_heads = tf.layers.Dense(embed_dim)

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

    def build_attention(self, inputs):
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

class TransformerBlock(tf.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.layers.Dense(ff_dim, activation=tf.nn.relu)
        self.ffn_output = tf.layers.Dense(embed_dim)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.layers.Dropout(rate)
        self.dropout2 = tf.layers.Dropout(rate)

    def build_transformer_block(self, inputs, training):
        attn_output = self.att.build_attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class VisionTransformer(tf.layers.Layer):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim):
        super(VisionTransformer, self).__init__()
        self.num_patches = num_patches
        embed_dim = embed_dim - embed_dim % num_heads
        self.d_embedding = tf.layers.Dense(embed_dim)
        self.pos_emb = self.add_variable("pos_emb", shape=[1, num_patches, embed_dim])
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

    def build_vision_transformer(self, x, training):
        x = self.d_embedding(x)
        x += self.pos_emb
        x = self.transformer_block.build_transformer_block(x, training=training)
        return x

class PatchMerging(tf.layers.Layer):
    def __init__(self,  dim, dim_scale=2):
        super(PatchMerging, self).__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.reduction = tf.layers.Dense(dim_scale * dim, use_bias=False)
        self.norm = LayerNormalization(epsilon=1e-6)

    def build_patch_merging(self, x):
        HW =int( np.sqrt(int(x.shape[1])))
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [B, HW, HW, C])

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, [B, -1, 4 * C])

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpansion(tf.layers.Layer):
    def __init__(self, dim, dim_scale=2):
        super(PatchExpansion, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = tf.layers.Dense(2 * dim, use_bias=False) if dim_scale == 2 else lambda x: x
        self.norm = LayerNormalization(epsilon=1e-6)

    def build_patch_expansion(self, x):
        #H, W = self.input_resolution
        HW =int( np.sqrt(int(x.shape[1])))
        x = self.expand(x)
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [B, HW, HW, C])
        x = tf.reshape(x, [B, HW, HW, -1, C // 4])
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        x = tf.reshape(x, [B, -1, HW * 2, C // 4])
        x = tf.reshape(x, [B, -1, C // 4])
        x = self.norm(x)
        return x

def VitImplementation(x, scope, numVit, embedDim, numHeads, ffDim, training):    
    this_input = x
    for ii in range(numVit):
        with tf.variable_scope(scope+"/Vit%02d" % (ii + 1)):
            vit = VisionTransformer(int(x.shape[1]), embedDim, numHeads, ffDim)
            vit_feature = vit.build_vision_transformer(x=this_input, training=training)
            this_input = vit_feature    
    return vit_feature

def PatchMergingImplementation(x, dim, dim_scale=2):
    merger = PatchMerging(dim=dim, dim_scale=dim_scale)
    result = merger.build_patch_merging(x=x)
    return result

def PatchExpandingImplementation(x, dim):
    expander = PatchExpansion(dim=dim)
    result = expander.build_patch_expansion(x=x)
    return result