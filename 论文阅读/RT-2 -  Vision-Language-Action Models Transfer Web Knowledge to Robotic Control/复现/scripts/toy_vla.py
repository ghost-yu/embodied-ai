"""教学用共享 Transformer；非 RT-2 官方模型。"""
import tensorflow as tf
V=272

class ToyVLA(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.vision=tf.keras.layers.Dense(64,activation='gelu')
        self.color=tf.keras.layers.Embedding(3,64)
        self.task=tf.keras.layers.Embedding(2,64)
        self.tokens=tf.keras.layers.Embedding(V,64)
        self.position=tf.keras.layers.Embedding(11,64)
        self.attn=tf.keras.layers.MultiHeadAttention(num_heads=4,key_dim=16,dropout=0)
        self.norm1=tf.keras.layers.LayerNormalization()
        self.ff=tf.keras.Sequential([tf.keras.layers.Dense(128,activation='gelu'),tf.keras.layers.Dense(64)])
        self.norm2=tf.keras.layers.LayerNormalization()
        self.head=tf.keras.layers.Dense(V)
    def call(self,inputs,training=False):
        image,color,task,previous=inputs
        prefix=tf.stack([self.vision(tf.reshape(image,[-1,12])),self.color(color),self.task(task)],axis=1)
        x=tf.concat([prefix,self.tokens(previous)],axis=1)
        x=x+self.position(tf.range(tf.shape(x)[1]))[None,:,:]
        x=self.norm1(x+self.attn(x,x,use_causal_mask=True,training=training))
        x=self.norm2(x+self.ff(x,training=training))
        return self.head(x[:,3:,:])
