import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Custom layers for Exp 10A model
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

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

class ClassToken(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.cls_token_weight = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )
        super(ClassToken, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_token = tf.broadcast_to(self.cls_token_weight, [batch_size, 1, self.embed_dim])
        return tf.concat([cls_token, inputs], axis=1)

def extract_attention_scores(model, X_sample):
    """
    To extract attention, we need to create a sub-model that goes up to the TransformerBlock,
    but we have to bypass the sequential functional API and call the layer directly to pass return_attention_scores=True.
    """
    # X_sample shape: (batch, 5, 2000, 1)
    
    # 1. Pass through CNN
    shared_cnn = model.get_layer('time_distributed').layer # or we can just pass through time_distributed
    time_distributed = model.get_layer('time_distributed')
    encoded = time_distributed(X_sample) # (batch, 5, 250, 64)
    
    # 2. Reshape
    reshape_layer = model.get_layer('reshape')
    unrolled = reshape_layer(encoded)
    
    # 3. Embedding
    embed_layer = None
    for layer in model.layers:
        if isinstance(layer, TokenAndPositionEmbedding):
            embed_layer = layer
            break
    embedded = embed_layer(unrolled)
    
    # 4. CLS Token
    cls_layer = None
    for layer in model.layers:
        if isinstance(layer, ClassToken):
            cls_layer = layer
            break
    sequence_with_cls = cls_layer(embedded)
    
    # 5. Transformer Block
    transformer_layer = None
    for layer in model.layers:
        if isinstance(layer, TransformerBlock):
            transformer_layer = layer
            break
            
    # Call transformer layer explicitly with return_attention_scores=True
    _, attn_scores = transformer_layer(sequence_with_cls, return_attention_scores=True)
    return attn_scores

def main():
    model_path = 'data/models/exp10a_model.keras'
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    
    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        print("Model or dataset missing.")
        return

    custom_objects = {
        'TransformerBlock': TransformerBlock,
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
        'ClassToken': ClassToken
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    data = np.load(dataset_path)
    X = data['X']
    Y = data['y']
    tics = data['tics']
    
    # Find samples
    # 1. Easy planet (just pick a random planet from train or test)
    # 2. Difficult planet (TIC 287328202_Positive which was recovered, or TIC 259377017_Positive which was missed)
    # Let's use TIC 287328202_Positive (Recovered in 10A)
    # 3. Noise (random)
    # 4. EB (random)
    
    # Let's get indexes
    planet_idx = np.where(Y == 1)[0][0]
    noise_idx = np.where(Y == 0)[0][0]
    eb_idx = np.where(Y == 2)[0][0]
    
    difficult_planet_str = "TIC TIC 287328202_Positive"
    difficult_idx = -1
    for i, tic_str in enumerate(tics):
        if tic_str == difficult_planet_str:
            difficult_idx = i
            break
            
    samples = {
        'Easy_Planet': X[planet_idx:planet_idx+1],
        'Difficult_Planet_287328202': X[difficult_idx:difficult_idx+1] if difficult_idx != -1 else None,
        'Noise': X[noise_idx:noise_idx+1],
        'EB': X[eb_idx:eb_idx+1]
    }
    
    import json
    results = {}
    
    for name, sample_x in samples.items():
        if sample_x is None:
            continue
        
        # Predict class just to see
        preds = model.predict(sample_x, verbose=0)
        pred_class = np.argmax(preds[0])
        
        # attn_scores shape: (batch, num_heads, query_seq, value_seq) -> (1, 4, 1251, 1251)
        attn_scores = extract_attention_scores(model, sample_x)
        
        # We care about what [CLS] attends to. [CLS] is at index 0 of query_seq.
        # So we want attn_scores[0, :, 0, 1:] (batch 0, all heads, query 0, values 1..1250)
        cls_attention = attn_scores[0, :, 0, 1:].numpy() # shape (4, 1250)
        
        # Average across the 4 heads to get an overall attention map
        mean_attention = np.mean(cls_attention, axis=0) # shape (1250,)
        
        # Sector averages
        sector_len = 250
        sector_attns = []
        for s in range(5):
            sector_attns.append(np.sum(mean_attention[s*sector_len:(s+1)*sector_len]))
            
        # Top 10 attended bins
        top_bins = np.argsort(mean_attention)[-10:][::-1]
        
        results[name] = {
            'predicted_class': int(pred_class),
            'sector_attention_sums': [float(s) for s in sector_attns],
            'top_attended_bins': [int(b) for b in top_bins],
            'top_attended_weights': [float(mean_attention[b]) for b in top_bins]
        }
        
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
