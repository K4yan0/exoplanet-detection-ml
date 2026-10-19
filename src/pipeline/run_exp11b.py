import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------------
# Core Custom Layers for Two-Stage Cross-Sector Transformer
# ---------------------------------------------------------------------------

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

    def call(self, inputs, training=False, attention_mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
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

# ---------------------------------------------------------------------------
# Two-Stage Model Builder
# ---------------------------------------------------------------------------

def build_two_stage_model():
    """
    Constructs the Exp 11A Two-Stage Cross-Sector Transformer with attention mask inputs.
    Block 1: Cross-Sector Mixing (S_i <-> S_j)
    Block 2: Readout Aggregation (CLS <-> S_i)
    """
    local_input = tf.keras.Input(shape=(5, 2000, 1), name='local_morphology_input')
    attn_mask_input_1 = tf.keras.Input(shape=(6, 6), name='attention_mask_input_1', dtype=tf.bool)
    attn_mask_input_2 = tf.keras.Input(shape=(6, 6), name='attention_mask_input_2', dtype=tf.bool)
    
    shared_cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(2000, 1)),
        tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 1000
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 500
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 250 spatial bins
    ], name='shared_high_res_cnn')
    
    encoded_sectors = tf.keras.layers.TimeDistributed(shared_cnn)(local_input) # (batch, 5, 250, 64)
    permuted = tf.keras.layers.Permute((2, 1, 3))(encoded_sectors) # (batch, 250, 5, 64)
    
    def flatten_batch_phase(x):
        batch = tf.shape(x)[0]
        return tf.reshape(x, [batch * 250, 5, 64])
    flattened_phases = tf.keras.layers.Lambda(flatten_batch_phase)(permuted)
    
    sector_cls = ClassToken(embed_dim=64, name='sector_cls')(flattened_phases) # (batch * 250, 6, 64)
    
    def tile_mask(x):
        batch = tf.shape(x)[0]
        mask_expanded = tf.expand_dims(x, 1) # (batch, 1, 6, 6)
        mask_tiled = tf.tile(mask_expanded, [1, 250, 1, 1]) # (batch, 250, 6, 6)
        return tf.reshape(mask_tiled, [batch * 250, 6, 6])
    
    broadcasted_mask_1 = tf.keras.layers.Lambda(tile_mask, output_shape=(None, 6, 6))(attn_mask_input_1)
    broadcasted_mask_2 = tf.keras.layers.Lambda(tile_mask, output_shape=(None, 6, 6))(attn_mask_input_2)
    
    # Block 1: Cross-Sector Mixing
    transformer_block_1 = TransformerBlockXAI(embed_dim=64, num_heads=4, ff_dim=128, name='cross_sector_transformer_1')
    cross_sector_att_1 = transformer_block_1(sector_cls, attention_mask=broadcasted_mask_1)
    
    # Block 2: Readout Aggregation
    transformer_block_2 = TransformerBlockXAI(embed_dim=64, num_heads=4, ff_dim=128, name='cross_sector_transformer_2')
    cross_sector_att_2 = transformer_block_2(cross_sector_att_1, attention_mask=broadcasted_mask_2)
    
    def extract_cls(x):
        return x[:, 0, :]
    phase_features_flat = tf.keras.layers.Lambda(extract_cls)(cross_sector_att_2) # (batch * 250, 64)
    
    def unflatten_batch_phase(x):
        batch = tf.shape(x)[0] // 250
        return tf.reshape(x, [batch, 250, 64])
    phase_features = tf.keras.layers.Lambda(unflatten_batch_phase)(phase_features_flat)
    
    # Global Phase Transformer
    global_cls = ClassToken(embed_dim=64, name='global_cls')(phase_features)
    phase_att = TransformerBlockXAI(embed_dim=64, num_heads=4, ff_dim=128, name='global_phase_transformer')(global_cls)
    final_features = tf.keras.layers.Lambda(extract_cls)(phase_att)
    
    # Classifier Head
    x = tf.keras.layers.Dropout(0.3)(final_features)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[local_input, attn_mask_input_1, attn_mask_input_2], outputs=outputs)
    return model

# ---------------------------------------------------------------------------
# Reliability Metrics: ECE & Brier Score
# ---------------------------------------------------------------------------

def calculate_ece(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return float(ece)

def calculate_brier(y_true, y_prob, n_classes=3):
    y_true_oh = to_categorical(y_true, num_classes=n_classes)
    return float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))

# ---------------------------------------------------------------------------
# Multi-Seed Execution Protocol (EXP11B)
# ---------------------------------------------------------------------------

DIFFICULT_TARGET_TICS = [
    '259377017', '36724087', '287328202', '345143460', '234994474',
    '150030205', '262530407', '181804752', '307809773', '254113311'
]

def run_seed_evaluation(seed, X_train, y_train, X_val, y_val, X_test, y_test, tics_test):
    print(f"\n{'='*60}")
    print(f"RUNNING EXP11B SEED: {seed}")
    print(f"{'='*60}")
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    full_mask = np.ones((6, 6), dtype=bool)
    ablated_mask = np.zeros((6, 6), dtype=bool)
    ablated_mask[0, :] = True # CLS reads all
    for i in range(1, 6):
        ablated_mask[i, 0] = True # Sector reads CLS
        ablated_mask[i, i] = True # Sector reads self only
        
    train_mask_1 = np.broadcast_to(full_mask, (X_train.shape[0], 6, 6))
    train_mask_2 = np.broadcast_to(full_mask, (X_train.shape[0], 6, 6))
    val_mask_1 = np.broadcast_to(full_mask, (X_val.shape[0], 6, 6))
    val_mask_2 = np.broadcast_to(full_mask, (X_val.shape[0], 6, 6))
    test_mask_1 = np.broadcast_to(full_mask, (X_test.shape[0], 6, 6))
    test_mask_2 = np.broadcast_to(full_mask, (X_test.shape[0], 6, 6))
    
    model = build_two_stage_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Balanced class weights
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
    )
    
    # Train
    history = model.fit(
        [X_train, train_mask_1, train_mask_2], y_train,
        validation_data=([X_val, val_mask_1, val_mask_2], y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # -----------------------------------------------------------------------
    # 1. INTACT EVALUATION (Tier 1: Classification & Tier 2: Reliability)
    # -----------------------------------------------------------------------
    # Standard forward pass
    intact_probs = model.predict([X_test, test_mask_1, test_mask_2], verbose=0)
    intact_preds = np.argmax(intact_probs, axis=1)
    
    acc = float(accuracy_score(y_test, intact_preds))
    y_test_oh = to_categorical(y_test, num_classes=3)
    try:
        roc_auc = float(roc_auc_score(y_test_oh, intact_probs, multi_class='ovr', average='macro'))
    except Exception:
        roc_auc = 0.5
        
    p_prec = float(precision_score(y_test == 1, intact_preds == 1, zero_division=0))
    p_rec = float(recall_score(y_test == 1, intact_preds == 1, zero_division=0))
    p_f1 = float(f1_score(y_test == 1, intact_preds == 1, zero_division=0))
    n_rec = float(recall_score(y_test == 0, intact_preds == 0, zero_division=0))
    eb_rec = float(recall_score(y_test == 2, intact_preds == 2, zero_division=0))
    cm = confusion_matrix(y_test, intact_preds).tolist()
    
    ece = calculate_ece(y_test, intact_probs)
    brier = calculate_brier(y_test, intact_probs)
    
    # Monte Carlo Dropout (50 passes) for epistemic uncertainty
    @tf.function(reduce_retracing=True)
    def mc_step(x, m1, m2):
        return model([x, m1, m2], training=True)
        
    mc_preds = [mc_step(X_test, test_mask_1, test_mask_2).numpy() for _ in range(50)]
    mc_preds = np.array(mc_preds) # (50, N, 3)
    mc_var_per_sample = np.mean(np.var(mc_preds, axis=0), axis=1) # Mean variance across classes
    mc_var_global = float(np.mean(mc_var_per_sample))

    
    # -----------------------------------------------------------------------
    # 2. TARGETED BEHAVIOR (Tier 3: 10 Difficult Validation Planets)
    # -----------------------------------------------------------------------
    targeted_results = {}
    recovered_count = 0
    for diff_id in DIFFICULT_TARGET_TICS:
        matches = [idx for idx, t in enumerate(tics_test) if diff_id in t and "Positive" in t]
        if len(matches) > 0:
            idx = matches[0]
            pred_cls = int(intact_preds[idx])
            p_conf = float(intact_probs[idx][1])
            p_var = float(mc_var_per_sample[idx])
            is_rec = bool(pred_cls == 1)
            if is_rec:
                recovered_count += 1
            targeted_results[diff_id] = {
                'in_test_set': True,
                'predicted_class': pred_cls,
                'p_planet': p_conf,
                'mc_variance': p_var,
                'recovered': is_rec
            }
        else:
            targeted_results[diff_id] = {'in_test_set': False}
            
    # -----------------------------------------------------------------------
    # 3. CAUSAL BEHAVIOR (Tier 4: Block 1 Cross-Sector Ablation)
    # -----------------------------------------------------------------------
    test_mask_1_ablated = np.broadcast_to(ablated_mask, (X_test.shape[0], 6, 6))
    ablated_probs = model.predict([X_test, test_mask_1_ablated, test_mask_2], verbose=0)
    ablated_preds = np.argmax(ablated_probs, axis=1)
    
    ablated_acc = float(accuracy_score(y_test, ablated_preds))
    ablated_p_rec = float(recall_score(y_test == 1, ablated_preds == 1, zero_division=0))
    ablated_p_f1 = float(f1_score(y_test == 1, ablated_preds == 1, zero_division=0))
    
    # Causal Drop
    acc_drop = acc - ablated_acc
    p_rec_drop = p_rec - ablated_p_rec
    
    print(f"Seed {seed} Summary:")
    print(f"  Intact Acc: {acc:.4f} | ROC-AUC: {roc_auc:.4f} | Planet F1: {p_f1:.4f} (P: {p_prec:.2f}, R: {p_rec:.2f})")
    print(f"  Reliability: ECE = {ece:.4f} | Brier = {brier:.4f} | MC-Var = {mc_var_global:.5f}")
    print(f"  Targeted Recovery: {recovered_count} difficult targets recovered")
    print(f"  Ablated Block 1 Acc: {ablated_acc:.4f} (Drop: {acc_drop:+.4f}) | Planet Recall: {ablated_p_rec:.4f} (Drop: {p_rec_drop:+.4f})")
    
    return {
        'seed': seed,
        'classification': {
            'accuracy': acc,
            'roc_auc_macro': roc_auc,
            'planet_precision': p_prec,
            'planet_recall': p_rec,
            'planet_f1': p_f1,
            'noise_recall': n_rec,
            'eb_recall': eb_rec,
            'confusion_matrix': cm
        },
        'reliability': {
            'ece': ece,
            'brier_score': brier,
            'mc_variance_global': mc_var_global
        },
        'targeted_recovery': {
            'recovered_count': recovered_count,
            'details': targeted_results
        },
        'causal_ablation': {
            'ablated_accuracy': ablated_acc,
            'ablated_planet_recall': ablated_p_rec,
            'ablated_planet_f1': ablated_p_f1,
            'accuracy_drop': acc_drop,
            'planet_recall_drop': p_rec_drop,
            'causal_pathway_functional': bool(p_rec_drop > 0.05 or acc_drop > 0.05)
        }
    }

def main():
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    if not os.path.exists(dataset_path):
        print(f"\n[ERROR] Preprocessed dataset not found at '{dataset_path}'.")
        print("To run this experiment on a new machine, you have two options:")
        print("  Option 1 (Fast): Copy the precompiled 'data/' directory manually into the project root.")
        print("  Option 2 (Re-acquire): Re-acquire the cohort from NASA MAST by running:")
        print("           python src/pipeline/acquire_exp8_cohort.py\n")
        return
        
    print(f"Loading 5-sector dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    tics = data['tics']
    
    # Defensive data sanitization for any missing telemetry bins
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Dataset Shape: {X.shape}, Labels Shape: {y.shape}")
    
    # 80/20 train/test split strictly identical to previous experiments
    X_train_val, X_test, y_train_val, y_test, tics_train_val, tics_test = train_test_split(
        X, y, tics, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_train_val, y_train_val, tics_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    seeds = [42, 100, 200, 300, 400]
    results = []
    
    for s in seeds:
        seed_res = run_seed_evaluation(s, X_train, y_train, X_val, y_val, X_test, y_test, tics_test)
        results.append(seed_res)
        
    # Compute aggregate summary statistics
    accs = [r['classification']['accuracy'] for r in results]
    precs = [r['classification']['planet_precision'] for r in results]
    recs = [r['classification']['planet_recall'] for r in results]
    f1s = [r['classification']['planet_f1'] for r in results]
    eces = [r['reliability']['ece'] for r in results]
    mc_vars = [r['reliability']['mc_variance_global'] for r in results]
    causal_drops = [r['causal_ablation']['planet_recall_drop'] for r in results]
    
    summary = {
        'seeds_evaluated': seeds,
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_planet_precision': float(np.mean(precs)),
        'std_planet_precision': float(np.std(precs)),
        'mean_planet_recall': float(np.mean(recs)),
        'std_planet_recall': float(np.std(recs)),
        'mean_planet_f1': float(np.mean(f1s)),
        'std_planet_f1': float(np.std(f1s)),
        'mean_ece': float(np.mean(eces)),
        'std_ece': float(np.std(eces)),
        'mean_mc_variance': float(np.mean(mc_vars)),
        'std_mc_variance': float(np.std(mc_vars)),
        'mean_causal_recall_drop': float(np.mean(causal_drops)),
        'per_seed_results': results
    }
    
    os.makedirs('docs/reports', exist_ok=True)
    report_path = 'docs/reports/EXP11B_multi_seed_evaluation.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "=" * 60)
    print("FINAL EXP11B MULTI-SEED STABILITY & CAUSAL REPORT")
    print("=" * 60)
    print(f"Mean Accuracy:       {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")
    print(f"Mean Planet Prec:    {summary['mean_planet_precision']:.4f} +/- {summary['std_planet_precision']:.4f}")
    print(f"Mean Planet Rec:     {summary['mean_planet_recall']:.4f} +/- {summary['std_planet_recall']:.4f}")
    print(f"Mean Planet F1:      {summary['mean_planet_f1']:.4f} +/- {summary['std_planet_f1']:.4f}")
    print(f"Mean ECE:            {summary['mean_ece']:.4f} +/- {summary['std_ece']:.4f}")
    print(f"Mean Causal Drop:    {summary['mean_causal_recall_drop']:.4f}")
    print(f"\nFull report written to: {report_path}")

if __name__ == '__main__':
    main()
