import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False, return_attention_scores=False):
        if return_attention_scores:
            attn_output, attn_scores = self.att(inputs, inputs, return_attention_scores=True)
        else:
            attn_output = self.att(inputs, inputs)
            attn_scores = None
            
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        if return_attention_scores:
            return self.layernorm2(out1 + ffn_output), attn_scores
        return self.layernorm2(out1 + ffn_output)

from src.pipeline.train_exp10c import SectorPhaseEmbedding, ClassToken

# Custom objects required to load the model
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'SectorPhaseEmbedding': SectorPhaseEmbedding,
    'ClassToken': ClassToken
}

def extract_attention():
    # 1. Load model with custom objects
    model_path = "data/models/exp10c_model.keras"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run training first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Trace the layers
    cnn_output = None
    embedding_output = None
    cls_token_output = None
    
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    tics = data['tics']
    
    # Find TIC TIC 259377017_Positive (one of the recovered targeted planets)
    target_tic = "TIC TIC 259377017_Positive"
    idx = np.where(tics == target_tic)[0]
    
    if len(idx) == 0:
        print(f"Target {target_tic} not found!")
        idx = np.where(y == 1)[0][0]
    else:
        idx = idx[0]
        
    X_sample = X[idx:idx+1]
    
    print(f"Extracting attention for {tics[idx]} (label={y[idx]})...")
    
    # Manual forward pass up to Transformer block
    # 1. Input
    x = X_sample
    
    # 2. TimeDistributed CNN
    td_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.TimeDistributed):
            td_layer = layer
            break
            
    if td_layer is None:
        print("TimeDistributed layer not found!")
        return
        
    x = td_layer(x)
    
    # Reshape (unroll)
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, (batch_size, 1250, 64))
    
    # 3. SectorPhaseEmbedding
    embed_layer = None
    for layer in model.layers:
        if isinstance(layer, SectorPhaseEmbedding):
            embed_layer = layer
            break
    
    if embed_layer is None:
        print("SectorPhaseEmbedding not found!")
        return
        
    x = embed_layer(x)
    
    # 4. ClassToken
    cls_layer = None
    for layer in model.layers:
        if isinstance(layer, ClassToken):
            cls_layer = layer
            break
            
    x = cls_layer(x)
    
    # 5. First TransformerBlock (get attention)
    transformer_layer = None
    for layer in model.layers:
        if isinstance(layer, TransformerBlock):
            transformer_layer = layer
            break
            
    _, attn_scores = transformer_layer(x, training=False, return_attention_scores=True)
    
    # attn_scores shape: (batch_size, num_heads, sequence_length, sequence_length)
    # The [CLS] token is at index 0 of sequence length (1 + 1250 = 1251)
    
    print("Attention scores shape:", attn_scores.shape)
    
    # Average across heads
    attn_scores_mean = tf.reduce_mean(attn_scores, axis=1) # (1, 1251, 1251)
    
    # Get [CLS] token attention (query = CLS token -> row 0)
    # The [CLS] token attends to the sequence
    cls_attention = attn_scores_mean[0, 0, 1:].numpy() # Skip self-attention to [CLS]
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    # Split into 5 sectors for easy viewing
    sector_len = 250
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for s in range(5):
        start = s * sector_len
        end = (s + 1) * sector_len
        plt.plot(range(start, end), cls_attention[start:end], color=colors[s], label=f'Sector {s+1}')
        
    plt.title(f'Exp 10C [CLS] Token Attention - {tics[idx]}')
    plt.xlabel('Flattened Bin Index')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'C:\Users\Admin\.gemini\antigravity-cli\brain\afbd7ad9-01de-4c5e-9ab9-cc1d18c908a6\EXP10C_attention_plot.png')
    print("Plot saved to EXP10C_attention_plot.png")

if __name__ == "__main__":
    extract_attention()
