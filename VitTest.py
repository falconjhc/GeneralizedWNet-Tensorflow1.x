import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Dense, LayerNormalization, Embedding, Flatten
from tensorflow.keras.models import Model

import numpy as np

tf.compat.v1.disable_eager_execution()  

# class MultiHeadSelfAttention(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads=8):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         if embed_dim % num_heads != 0:
#             raise ValueError(f'divide error')
#         self.projection_dim = embed_dim // num_heads
#         self.query_dense = Dense(embed_dim)
#         self.key_dense = Dense(embed_dim)
#         self.value_dense = Dense(embed_dim)
#         self.combine_heads = Dense(embed_dim)

#     def attention(self, query, key, value):
#         score = tf.matmul(query, key, transpose_b=True)
#         dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
#         scaled_score = score / tf.math.sqrt(dim_key)
#         weights = tf.nn.softmax(scaled_score, axis=-1)
#         output = tf.matmul(weights, value)
#         return output, weights

#     def separate_heads(self, x, batch_size):
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
#         return tf.transpose(x, perm=[0, 2, 1, 3])

#     def Build(self, inputs):
#         batch_size = tf.shape(inputs)[0]
#         query = self.query_dense(inputs)
#         key = self.key_dense(inputs)
#         value = self.value_dense(inputs)

#         query = self.separate_heads(query, batch_size)
#         key = self.separate_heads(key, batch_size)
#         value = self.separate_heads(value, batch_size)

#         attention, _ = self.attention(query, key, value)
#         attention = tf.transpose(attention, perm=[0, 2, 1, 3])
#         concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
#         output = self.combine_heads(concat_attention)
#         return output

# class TransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = MultiHeadSelfAttention(embed_dim, num_heads)
#         self.ffn = tf.keras.Sequential(
#             [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
#         )
#         self.layernorm1 = LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = LayerNormalization(epsilon=1e-6)
#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)

#     def Build(self, inputs, training):
#         attn_output = self.att.Build(inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)

# class VisionTransformer(Model):
#     def __init__(self, num_patches, embed_dim, num_heads, ff_dim):
#         super(VisionTransformer, self).__init__()
#         self.num_patches = num_patches
#         self.d_embedding = Dense(embed_dim)
#         self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches, embed_dim))
#         self.class_emb = self.add_weight("class_emb", shape=(1, 1, embed_dim))
#         self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
#         self.flatten = Flatten()

#     def Build(self, x, training):
#         x = self.d_embedding(x)
#         x += self.pos_emb
#         x = self.transformer_block.Build(x, training=training)
#         return x

# def main():
#     batchSize=8
#     imgW=64
#     imgH=64
#     imgC=1
#     patchSize = 4 
#     num_patches = (imgW // patchSize) ** 2  
#     # num_patches = 64  
#     embed_dim = 96  
#     num_heads = 3  
#     ff_dim = 96*4  


#     # Original inpit
#     x_train = np.random.random((batchSize, imgW, imgH, imgC))
#     x_patches = tf.image.extract_patches(
#         images=tf.constant(x_train, dtype=tf.float32),
#         sizes=[1, patchSize, patchSize, 1],
#         strides=[1, patchSize, patchSize, 1],
#         rates=[1, 1, 1, 1],
#         padding='VALID'
#     )
#     batch_size, rows, cols, channels = x_patches.shape
#     x_patches = tf.reshape(x_patches, (batch_size, rows * cols, patchSize * patchSize * imgC))
#     print("Shape of patches:", x_patches.shape)
#     vit = VisionTransformer(num_patches, embed_dim, num_heads, ff_dim)
#     output = vit.Build(x_patches, training=False)
#     output=tf.reshape(output, (batch_size, imgW, imgH, int(output.shape[-1]//(patchSize*patchSize)))) # ???
#     print("Model output shape:", output.shape)

# if __name__ == "__main__":
#     main()



# import tensorflow as tf
# from tensorflow.keras.layers import Dense, LayerNormalization, Embedding, Flatten
# from tensorflow.keras.models import Model

# import numpy as np

# tf.compat.v1.disable_eager_execution()  

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

    def Build(self, inputs):
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
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def Build(self, inputs, training):
        attn_output = self.att.Build(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class VisionTransformer(Model):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim, num_classes):
        super(VisionTransformer, self).__init__()
        self.num_patches = num_patches
        self.d_embedding = Dense(embed_dim)
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches + 1, embed_dim))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, embed_dim))
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.flatten = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def Build(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.d_embedding(x)
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.class_emb.shape[2]])
        x = tf.concat([class_emb, x], axis=1)
        x += self.pos_emb
        x = self.transformer_block.Build(x, training=training)
        x = self.flatten(x[:, 0])
        x = self.fc(x)
        return x

def main():

    patch_size = 16  
    num_patches = (224 // patch_size) ** 2  
    # num_patches = 64  
    embed_dim = 256  
    num_heads = 8  
    ff_dim = 512  
    num_classes = 10  


    # Original inpit
    x_train = np.random.random((4, 224, 224, 3))
    x_patches = tf.image.extract_patches(
        images=tf.constant(x_train, dtype=tf.float32),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    batch_size, rows, cols, channels = x_patches.shape
    x_patches = tf.reshape(x_patches, (batch_size, rows * cols, patch_size * patch_size * 3))
    print("Shape of patches:", x_patches.shape)
    # Patch
    # x_train = np.random.random((2, num_patches, 16*16*3))


    vit = VisionTransformer(num_patches, embed_dim, num_heads, ff_dim, num_classes)
    output = vit.Build(x_patches, training=False)
    # vit.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # output = vit.predict(x_train)
    # output = vit.predict(x_patches)
    # output = vit.predict(x_patches, steps=1)
    print("Model output shape:", output.shape)

if __name__ == "__main__":
    main()
