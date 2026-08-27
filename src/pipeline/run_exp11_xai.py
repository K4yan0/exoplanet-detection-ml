import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class TransformerBlockXAI(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlockXAI, self).__init__(**kwargs)
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
            attn_output, attention_scores = self.att(inputs, inputs, return_attention_scores=True)
        else:
            attn_output = self.att(inputs, inputs)
            
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        if return_attention_scores:
            return self.layernorm2(out1 + ffn_output), attention_scores
        return self.layernorm2(out1 + ffn_output)

class ClassToken(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.cls_token_weight = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='random_normal',
            trainable=True,
            name=self.name + '_weight'
        )
        super(ClassToken, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_token = tf.broadcast_to(self.cls_token_weight, [batch_size, 1, self.embed_dim])
        return tf.concat([cls_token, inputs], axis=1)

def build_exp11_xai_model():
    local_input = tf.keras.Input(shape=(5, 2000, 1), name='local_morphology_input')
    
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
    
    permuted = tf.keras.layers.Permute((2, 1, 3))(encoded_sectors)
    
    def flatten_batch_phase(x):
        batch = tf.shape(x)[0]
        return tf.reshape(x, [batch * 250, 5, 64])
    flattened_phases = tf.keras.layers.Lambda(flatten_batch_phase)(permuted)
    
    sector_cls = ClassToken(embed_dim=64, name='sector_cls')(flattened_phases) # (batch * 250, 6, 64)
    
    # We use our XAI block to return attention scores
    transformer_block = TransformerBlockXAI(embed_dim=64, num_heads=4, ff_dim=128, name='cross_sector_transformer')
    cross_sector_att, attention_scores = transformer_block(sector_cls, return_attention_scores=True)
    
    def extract_cls(x):
        return x[:, 0, :]
    phase_features_flat = tf.keras.layers.Lambda(extract_cls)(cross_sector_att) # (batch * 250, 64)
    
    def unflatten_batch_phase(x):
        batch = tf.shape(x)[0] // 250
        return tf.reshape(x, [batch, 250, 64])
    phase_features = tf.keras.layers.Lambda(unflatten_batch_phase)(phase_features_flat)
    
    global_cls = ClassToken(embed_dim=64, name='global_cls')(phase_features)
    phase_att = TransformerBlockXAI(embed_dim=64, num_heads=4, ff_dim=128, name='global_phase_transformer')(global_cls)
    final_features = tf.keras.layers.Lambda(extract_cls)(phase_att)
    
    x = tf.keras.layers.Dropout(0.3)(final_features)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    # We output both the classification result AND the attention scores!
    # Attention scores shape: (batch * 250, num_heads, 6, 6)
    model = tf.keras.Model(inputs=local_input, outputs=[outputs, attention_scores])
    return model

def train_xai_seed(seed):
    print(f"\\n{'='*50}\\nTraining Exp 11 XAI with SEED={seed}\\n{'='*50}")
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    tics = data['tics']
    
    X_train_val, X_test, y_train_val, y_test, tics_train_val, tics_test = train_test_split(
        X, y, tics, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, tics_train, tics_val = train_test_split(
        X_train_val, y_train_val, tics_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    model = build_exp11_xai_model()
    
    # Since we added attention scores to outputs, we need a custom loss or specify loss just for output 0
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=['sparse_categorical_crossentropy', None],
        metrics=[['accuracy'], None]
    )
    
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min'
        )
    ]
    
    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )
    
    return model, X, y, tics

def plot_attention(attention_scores, title, filename):
    # attention_scores shape: (250, num_heads, 6, 6)
    # We want to look at the attention from the CLS token to the 5 sectors
    # CLS token is index 0. Sectors are indices 1 to 5.
    
    # Let's average across heads, and across phases, to get a macroscopic view
    # Average across all 250 phases and 4 heads
    mean_attention = np.mean(attention_scores, axis=(0, 1)) # Shape: (6, 6)
    
    labels = ['CLS', 'S1', 'S2', 'S3', 'S4', 'S5']
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_attention, xticklabels=labels, yticklabels=labels, cmap='viridis', annot=True, fmt=".3f")
    plt.title(f"Mean Attention Matrix over Phase Bins - {title}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    # Train Seed 300 (Successful)
    model_300, X_data, y_data, tics_data = train_xai_seed(300)
    
    # Train Seed 100 (Collapsed)
    model_100, _, _, _ = train_xai_seed(100)
    
    # Let's find specific cases
    # A. Target Difficult Planet
    target_tic = "TIC TIC 259377017_Positive" # We know this is a difficult target
    target_idx = np.where(tics_data == target_tic)[0][0]
    
    # B. Clean Planet (e.g., standard easy planet from training set, label 1)
    clean_planet_idx = np.where((y_data == 1) & (tics_data != target_tic))[0][0]
    
    # C. EB (label 2)
    eb_idx = np.where(y_data == 2)[0][0]
    
    # D. Noise FP (label 0)
    noise_idx = np.where(y_data == 0)[0][0]
    
    def generate_diagnostics(model, seed_str):
        cases = [
            (target_idx, "Difficult Target (TIC 259377017)"),
            (clean_planet_idx, "Clean Planet"),
            (eb_idx, "Eclipsing Binary"),
            (noise_idx, "Noise False Positive")
        ]
        
        for idx, title in cases:
            x_sample = X_data[idx:idx+1]
            probs, att = model.predict(x_sample, verbose=0)
            # att shape: (batch*250, 4, 6, 6) -> (250, 4, 6, 6)
            plot_attention(att, f"{title} (Seed {seed_str})", f"exp11_att_{seed_str}_{title.replace(' ', '_').replace('(', '').replace(')', '')}.png")
            print(f"[{seed_str}] {title} predicted class: {np.argmax(probs[0])} (True: {y_data[idx]})")

    print("Generating XAI for Seed 300 (Successful)...")
    generate_diagnostics(model_300, "300")
    
    print("Generating XAI for Seed 100 (Collapsed)...")
    generate_diagnostics(model_100, "100")
