import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.pipeline.train_exp10c import build_exp10c_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss, confusion_matrix

def train_and_evaluate_seed(seed):
    print(f"\n{'='*50}\nStarting Exp 10E with SEED={seed}\n{'='*50}")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # 1. Load Data
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    tics = data['tics']
    
    # Stratified split - STRICTLY IDENTICAL TO 10C (random_state=42)
    X_train_val, X_test, y_train_val, y_test, tics_train_val, tics_test = train_test_split(
        X, y, tics, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, tics_train, tics_val = train_test_split(
        X_train_val, y_train_val, tics_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 2. Build Model
    model = build_exp10c_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Train
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
    
    # 4. Evaluate (simplified for script)
    print("Evaluating...")
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    
    # Planet metrics (Planet is class 1)
    y_test_binary_planet = (y_test == 1).astype(int)
    preds_binary_planet = (preds == 1).astype(int)
    
    planet_precision = precision_score(y_test_binary_planet, preds_binary_planet, zero_division=0)
    planet_recall = recall_score(y_test_binary_planet, preds_binary_planet, zero_division=0)
    
    # Noise metrics (Noise is class 0)
    y_test_binary_noise = (y_test == 0).astype(int)
    preds_binary_noise = (preds == 0).astype(int)
    noise_recall = recall_score(y_test_binary_noise, preds_binary_noise, zero_division=0)
    
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Diagnostic recovery
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
        
        # Simple prediction without MC dropout to save time
        target_prob = model.predict(X_sample, verbose=0)[0]
        pred_class = np.argmax(target_prob)
        if pred_class == 1:
            recovered += 1
            
    print(f"Seed {seed} completed -> Acc: {acc:.4f}, Planet Prec: {planet_precision:.4f}, Planet Rec: {planet_recall:.4f}, Targeted: {recovered}/10")
    
    return {
        'seed': seed,
        'accuracy': acc,
        'planet_precision': planet_precision,
        'planet_recall': planet_recall,
        'noise_recall': noise_recall,
        'targeted_recovery': recovered
    }

if __name__ == '__main__':
    seeds = [42, 100, 200, 300, 400]
    results = []
    
    for seed in seeds:
        res = train_and_evaluate_seed(seed)
        results.append(res)
        
    print("\n\n" + "="*50)
    print("FINAL STABILITY REPORT (Exp 10E)")
    print("="*50)
    
    for r in results:
        print(f"Seed {r['seed']:3d} | Acc: {r['accuracy']:.4f} | Prec: {r['planet_precision']:.4f} | Rec: {r['planet_recall']:.4f} | Target: {r['targeted_recovery']}/10")
    
    accs = [r['accuracy'] for r in results]
    precs = [r['planet_precision'] for r in results]
    recs = [r['planet_recall'] for r in results]
    
    print("\nSummary Statistics:")
    print(f"Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    print(f"Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
