import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss, roc_auc_score
from tensorflow.keras.utils import to_categorical

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

def build_exp11_model():
    """
    Exp 11: Phase-Aligned Cross-Sector Self-Attention
    """
    local_input = tf.keras.Input(shape=(5, 2000, 1), name='local_morphology_input')
    
    # 1. Shared CNN
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
    
    # 2. Reshape to group by phase bin
    # We want (batch, 250, 5, 64)
    permuted = tf.keras.layers.Permute((2, 1, 3))(encoded_sectors)
    
    # Reshape to (batch * 250, 5, 64) to apply standard Transformer across sectors
    def flatten_batch_phase(x):
        batch = tf.shape(x)[0]
        return tf.reshape(x, [batch * 250, 5, 64])
    flattened_phases = tf.keras.layers.Lambda(flatten_batch_phase)(permuted)
    
    # 3. Add Phase-Level CLS token to aggregate sector information
    sector_cls = ClassToken(embed_dim=64, name='sector_cls')(flattened_phases) # (batch * 250, 6, 64)
    
    # 4. Cross-Sector Self-Attention
    # This attends ONLY across the 5 sectors (plus the CLS token) for a specific phase
    cross_sector_att = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, name='cross_sector_transformer')(sector_cls)
    
    # 5. Extract Sector CLS token output
    def extract_cls(x):
        return x[:, 0, :]
    phase_features_flat = tf.keras.layers.Lambda(extract_cls)(cross_sector_att) # (batch * 250, 64)
    
    # 6. Reshape back to (batch, 250, 64) - Now we have 250 aggregated phase features
    def unflatten_batch_phase(x):
        batch = tf.shape(x)[0] // 250
        return tf.reshape(x, [batch, 250, 64])
    phase_features = tf.keras.layers.Lambda(unflatten_batch_phase)(phase_features_flat)
    
    # 7. Learned Aggregation over the 250 phases using a Global CLS token
    global_cls = ClassToken(embed_dim=64, name='global_cls')(phase_features) # (batch, 251, 64)
    phase_att = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, name='global_phase_transformer')(global_cls)
    final_features = tf.keras.layers.Lambda(extract_cls)(phase_att) # (batch, 64)
    
    # 8. Classifier
    x = tf.keras.layers.Dropout(0.3)(final_features)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=local_input, outputs=outputs)
    return model

def train_and_evaluate_seed(seed):
    print(f"\n{'='*50}\nStarting Exp 11 with SEED={seed}\n{'='*50}")
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    tics = data['tics']
    
    # Stratified split - STRICTLY IDENTICAL TO 10E
    X_train_val, X_test, y_train_val, y_test, tics_train_val, tics_test = train_test_split(
        X, y, tics, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, tics_train, tics_val = train_test_split(
        X_train_val, y_train_val, tics_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    model = build_exp11_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ]
    
    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )
    
    print(f"Training stopped at epoch {len(history.epoch)}")
    
    print("Evaluating...")
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    # Core Metrics
    acc = accuracy_score(y_test, preds)
    y_test_oh = to_categorical(y_test, num_classes=3)
    roc_auc = roc_auc_score(y_test_oh, probs, multi_class='ovr')
    
    # Planet metrics (Planet is class 1)
    y_test_binary_planet = (y_test == 1).astype(int)
    preds_binary_planet = (preds == 1).astype(int)
    
    planet_precision = precision_score(y_test_binary_planet, preds_binary_planet, zero_division=0)
    planet_recall = recall_score(y_test_binary_planet, preds_binary_planet, zero_division=0)
    planet_f1 = f1_score(y_test_binary_planet, preds_binary_planet, zero_division=0)
    
    # Noise metrics (Noise is class 0)
    noise_recall = recall_score((y_test == 0).astype(int), (preds == 0).astype(int), zero_division=0)
    
    # EB metrics (EB is class 2)
    eb_recall = recall_score((y_test == 2).astype(int), (preds == 2).astype(int), zero_division=0)
    
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Targeted recovery
    target_tics = [
        "TIC TIC 259377017_Positive",
        "TIC TIC 36724087_Positive",
        "TIC TIC 287328202_Positive",
        "TIC TIC 345143460_Positive",
        "TIC TIC 234994474_Positive",
        "TIC TIC 150030205_Positive",
        "TIC TIC 262530407_Positive",
        "TIC TIC 181804752_Positive",
        "TIC TIC 307809773_Positive",
        "TIC TIC 254113311_Positive"
    ]
    
    recovered = 0
    for target in target_tics:
        idx = np.where(tics == target)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        X_sample = X[idx:idx+1]
        target_prob = model.predict(X_sample, verbose=0)[0]
        if np.argmax(target_prob) == 1:
            recovered += 1
            
    print(f"Seed {seed} -> Acc: {acc:.4f}, Prec: {planet_precision:.4f}, Rec: {planet_recall:.4f}, Target: {recovered}/10")
    
    return {
        'seed': seed,
        'accuracy': acc,
        'planet_precision': planet_precision,
        'planet_recall': planet_recall,
        'planet_f1': planet_f1,
        'noise_recall': noise_recall,
        'eb_recall': eb_recall,
        'roc_auc': roc_auc,
        'targeted_recovery': recovered
    }

if __name__ == '__main__':
    seeds = [42, 100, 200, 300, 400]
    results = []
    
    for seed in seeds:
        res = train_and_evaluate_seed(seed)
        results.append(res)
        
    print("\n\n" + "="*50)
    print("FINAL STABILITY REPORT (Exp 11)")
    print("="*50)
    
    for r in results:
        print(f"Seed {r['seed']:3d} | Acc: {r['accuracy']:.4f} | Prec: {r['planet_precision']:.4f} | Rec: {r['planet_recall']:.4f} | F1: {r['planet_f1']:.4f} | Target: {r['targeted_recovery']}/10")
    
    accs = [r['accuracy'] for r in results]
    precs = [r['planet_precision'] for r in results]
    recs = [r['planet_recall'] for r in results]
    f1s = [r['planet_f1'] for r in results]
    
    print("\nSummary Statistics:")
    print(f"Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    print(f"Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    print(f"F1 Score:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
