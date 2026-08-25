import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

RANDOM_SEED = 42

def compute_ece(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def enable_dropout(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.training = True
        elif isinstance(layer, tf.keras.layers.TimeDistributed):
            for inner_layer in layer.layer.layers:
                if isinstance(inner_layer, tf.keras.layers.Dropout):
                    inner_layer.training = True
    return model

def main():
    print("--- Exp 8 Full Evaluation Matrix ---")
    
    data = np.load('data/tess_ml_arrays/tess_dataset_exp8.npz')
    X = data['X']
    Y = data['y']
    TICS = data['tics']
    
    # We must use exactly the same train/test split to evaluate on the test set
    _, X_test, _, y_test, _, tics_test = train_test_split(X, Y, TICS, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
    
    model = tf.keras.models.load_model('data/models/exp8_model.keras')
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    
    print("\n1. Overall Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Noise', 'Planet', 'EB']))
    
    print("\n2. Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # ROC AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    macro_roc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
    print(f"\nMacro ROC-AUC: {macro_roc:.4f}")
    
    # Brier Score (Planet class)
    brier = brier_score_loss((y_test == 1).astype(int), y_prob[:, 1])
    print(f"Brier Score (Planet): {brier:.4f}")
    
    # ECE
    ece = compute_ece(y_test, y_prob)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # MC-Dropout for the 10 targeted planets
    print("\n--- MC-Dropout Uncertainty for 10 Targeted Planets ---")
    mc_model = enable_dropout(model)
    
    missed_tics = [
        'TIC TIC 259377017_Positive', 'TIC TIC 36724087_Positive',
        'TIC TIC 287328202_Positive', 'TIC TIC 345143460_Positive',
        'TIC TIC 234994474_Positive', 'TIC TIC 150030205_Positive',
        'TIC TIC 262530407_Positive', 'TIC TIC 181804752_Positive',
        'TIC TIC 307809773_Positive', 'TIC TIC 254113311_Positive'
    ]
    
    n_iterations = 50
    for tic in missed_tics:
        idx = np.where(TICS == tic)[0]
        if len(idx) == 0: continue
        
        idx = idx[0]
        x_input = X[idx:idx+1]
        
        mc_preds = []
        for _ in range(n_iterations):
            preds = mc_model(x_input, training=True)
            mc_preds.append(preds[0][1].numpy()) # P(Planet)
            
        mc_preds = np.array(mc_preds)
        mean_p = np.mean(mc_preds)
        std_p = np.std(mc_preds)
        
        print(f"{tic}: P(Planet) = {mean_p:.3f} ± {std_p:.3f} (Min: {np.min(mc_preds):.3f}, Max: {np.max(mc_preds):.3f})")

if __name__ == '__main__':
    main()
