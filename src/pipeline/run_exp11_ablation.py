import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

    def call(self, inputs, training=False, return_attention_scores=False, attention_mask=None):
        if return_attention_scores:
            attn_output, attention_scores = self.att(inputs, inputs, attention_mask=attention_mask, return_attention_scores=True)
        else:
            attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
            
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

def build_exp11_model_with_mask():
    local_input = tf.keras.Input(shape=(5, 2000, 1), name='local_morphology_input')
    attn_mask_input = tf.keras.Input(shape=(6, 6), name='attention_mask_input', dtype=tf.bool)
    
    shared_cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2000, 1)),
        tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 1000
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 500
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 250
    ], name='shared_high_res_cnn')
    
    encoded_sectors = tf.keras.layers.TimeDistributed(shared_cnn)(local_input) # (batch, 5, 250, 64)
    permuted = tf.keras.layers.Permute((2, 1, 3))(encoded_sectors)
    
    def flatten_batch_phase(x):
        batch = tf.shape(x)[0]
        return tf.reshape(x, [batch * 250, 5, 64])
    flattened_phases = tf.keras.layers.Lambda(flatten_batch_phase)(permuted)
    
    sector_cls = ClassToken(embed_dim=64, name='sector_cls')(flattened_phases) # (batch * 250, 6, 64)
    
    # Broadcast the mask to match batch * 250 size
    def tile_mask(x):
        mask, features = x
        # mask is (batch, 6, 6)
        batch = tf.shape(mask)[0]
        # we want (batch * 250, 6, 6). 
        mask_expanded = tf.expand_dims(mask, 1) # (batch, 1, 6, 6)
        mask_tiled = tf.tile(mask_expanded, [1, 250, 1, 1]) # (batch, 250, 6, 6)
        return tf.reshape(mask_tiled, [batch * 250, 6, 6])
    
    broadcasted_mask = tf.keras.layers.Lambda(tile_mask)([attn_mask_input, sector_cls])
    
    transformer_block = TransformerBlockXAI(embed_dim=64, num_heads=4, ff_dim=128, name='cross_sector_transformer')
    cross_sector_att = transformer_block(sector_cls, attention_mask=broadcasted_mask)
    
    def extract_cls(x):
        return x[:, 0, :]
    phase_features_flat = tf.keras.layers.Lambda(extract_cls)(cross_sector_att)
    
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
    
    model = tf.keras.Model(inputs=[local_input, attn_mask_input], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("Loading datasets...")
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    # Base full attention mask (True allows attention, False masks it out)
    full_mask = np.ones((6, 6), dtype=bool)
    
    # Train normal Seed 300 model (mask = True everywhere)
    tf.random.set_seed(300)
    np.random.seed(300)
    
    print("\nTraining Seed 300 with full attention...")
    model = build_exp11_model_with_mask()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Generate batched masks for training
    train_mask_batch = np.broadcast_to(full_mask, (X_train.shape[0], 6, 6))
    val_mask_batch = np.broadcast_to(full_mask, (X_val.shape[0], 6, 6))
    
    model.fit(
        [X_train, train_mask_batch], y_train,
        validation_data=([X_val, val_mask_batch], y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    print("\nEvaluating Intact Model...")
    val_loss, val_acc = model.evaluate([X_val, val_mask_batch], y_val, verbose=0)
    print(f"Intact Validation Accuracy: {val_acc:.4f}")
    
    # Find targeted difficult planets
    # Planet class is 1. We know from prior scripts some planets are recovered 10/10.
    planet_indices = np.where(y_val == 1)[0]
    intact_preds = model.predict([X_val, val_mask_batch], verbose=0)
    intact_planet_recall = np.mean(np.argmax(intact_preds[planet_indices], axis=1) == 1)
    print(f"Intact Planet Recall: {intact_planet_recall:.4f}")
    
    print("\nEvaluating Ablated Model (NO CROSS-SECTOR ATTENTION)...")
    # Ablated mask: 
    # Row 0 (CLS): [1, 1, 1, 1, 1, 1] (CLS reads from all)
    # Row 1 (S1): [1, 1, 0, 0, 0, 0] (S1 reads CLS, S1)
    # Row 2 (S2): [1, 0, 1, 0, 0, 0]
    # Row 3 (S3): [1, 0, 0, 1, 0, 0]
    # Row 4 (S4): [1, 0, 0, 0, 1, 0]
    # Row 5 (S5): [1, 0, 0, 0, 0, 1]
    ablated_mask = np.zeros((6, 6), dtype=bool)
    ablated_mask[0, :] = True
    for i in range(1, 6):
        ablated_mask[i, 0] = True
        ablated_mask[i, i] = True
        
    ablated_mask_batch = np.broadcast_to(ablated_mask, (X_val.shape[0], 6, 6))
    
    ab_loss, ab_acc = model.evaluate([X_val, ablated_mask_batch], y_val, verbose=0)
    print(f"Ablated Validation Accuracy: {ab_acc:.4f}")
    
    ab_preds = model.predict([X_val, ablated_mask_batch], verbose=0)
    ab_planet_recall = np.mean(np.argmax(ab_preds[planet_indices], axis=1) == 1)
    print(f"Ablated Planet Recall: {ab_planet_recall:.4f}")

    print("\nEvaluating CLS-Only Ablation (SECTORS CAN'T READ FROM EACH OTHER OR CLS)...")
    cls_only_mask = np.zeros((6, 6), dtype=bool)
    cls_only_mask[0, :] = True
    for i in range(1, 6):
        cls_only_mask[i, i] = True
    cls_only_mask_batch = np.broadcast_to(cls_only_mask, (X_val.shape[0], 6, 6))
    
    cls_loss, cls_acc = model.evaluate([X_val, cls_only_mask_batch], y_val, verbose=0)
    print(f"CLS-Only Ablation Validation Accuracy: {cls_acc:.4f}")
    cls_preds = model.predict([X_val, cls_only_mask_batch], verbose=0)
    cls_planet_recall = np.mean(np.argmax(cls_preds[planet_indices], axis=1) == 1)
    print(f"CLS-Only Ablation Planet Recall: {cls_planet_recall:.4f}")
    
    print("\nEvaluating Extreme Ablation (CLS CAN ONLY READ ITSELF, SECTORS READ THEMSELVES)...")
    extreme_mask = np.eye(6, dtype=bool)
    extreme_mask_batch = np.broadcast_to(extreme_mask, (X_val.shape[0], 6, 6))
    ext_loss, ext_acc = model.evaluate([X_val, extreme_mask_batch], y_val, verbose=0)
    print(f"Extreme Ablation Validation Accuracy: {ext_acc:.4f}")

if __name__ == "__main__":
    main()
