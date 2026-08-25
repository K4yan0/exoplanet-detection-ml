import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

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

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
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

def build_exp10_model():
    """
    Exp 10 Architecture (High-Resolution Cross-Attention Fusion)
    
    1. Local High-Resolution Encoder (Shared CNN)
    2. Sequence Unrolling (Concatenating the 5 sectors into a single high-resolution sequence)
    3. Positional Encoding
    4. Self-Attention (Allows bins from Sector 1 to explicitly attend to bins in Sector 5)
    5. Global Pooling & Classification
    """
    
    local_input = tf.keras.Input(shape=(5, 2000, 1), name='local_morphology_input')
    
    # 1. High-Resolution Encoder
    # We pool gently to reduce the sequence length to a manageable size for attention, 
    # but we DO NOT crush it down to 20 bins like Exp 9C. We preserve 250 spatial bins per sector.
    shared_cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2000, 1)),
        tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 1000
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 500
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 250 spatial bins
    ], name='shared_high_res_cnn')
    
    encoded_sectors = tf.keras.layers.TimeDistributed(shared_cnn)(local_input) # (batch, 5, 250, 64)
    
    # 2. Sequence Unrolling
    # Shape becomes (batch, 1250, 64)
    # This creates a single continuous timeline of high-resolution local features
    unrolled_sequence = tf.keras.layers.Reshape((1250, 64))(encoded_sectors)
    
    # 3. Positional Encoding
    # The network must know which bin it is looking at to calculate phase drift
    embedded_sequence = TokenAndPositionEmbedding(maxlen=1250, embed_dim=64)(unrolled_sequence)
    
    # 4. Cross-Sector Relational Fusion (Self-Attention)
    # This explicitly compares the high-res morphology of Sector A with Sector B
    x = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)(embedded_sequence)
    
    # 5. Global Pooling & Classifier
    # Since the Transformer has already mixed the temporal and spatial information, 
    # we can now safely pool without losing relational context.
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=local_input, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print(f"\n==================================================")
    print(f"TRAINING EXP 10: High-Resolution Cross-Attention")
    print(f"==================================================")
    
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
        
    data = np.load(dataset_path)
    X = data['X']
    Y = data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    model = build_exp10_model()
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f}")
    
    os.makedirs('data/models', exist_ok=True)
    save_path = 'data/models/exp10_model.keras'
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
