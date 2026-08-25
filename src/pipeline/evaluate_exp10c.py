import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from train_exp10c import TransformerBlock, SectorPhaseEmbedding, ClassToken

RANDOM_SEED = 42

# We need the enable_dropout function for MC-Dropout
def enable_dropout(model):
    """Enable dropout layers during test time for MC-Dropout"""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.training = True
        elif hasattr(layer, 'layers'):
            enable_dropout(layer)
        # Handle the TransformerBlock which has internal dropouts
        elif isinstance(layer, TransformerBlock):
            layer.dropout1.training = True
            layer.dropout2.training = True

def predict_with_uncertainty(model, X, num_samples=30):
    """Run MC-Dropout for uncertainty estimation"""
    predictions = []
    for _ in range(num_samples):
        predictions.append(model.predict(X, verbose=0))
    
    predictions = np.array(predictions) # (num_samples, batch_size, num_classes)
    means = np.mean(predictions, axis=0)
    stds = np.std(predictions, axis=0)
    return means, stds

def evaluate_model():
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    model_path = 'data/models/exp10c_model.keras'
    
    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        print("Missing dataset or model file.")
        return

    custom_objects = {
        'TransformerBlock': TransformerBlock,
        'SectorPhaseEmbedding': SectorPhaseEmbedding,
        'ClassToken': ClassToken
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    data = np.load(dataset_path)
    X = data['X']
    Y = data['y']
    tics = data['tics']
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y
    )
    
    _, tics_test = train_test_split(
        tics, test_size=0.2, random_state=RANDOM_SEED, stratify=Y
    )

    print("\n--- Exp 10C (Sector-Aware CLS Transformer) Full Evaluation ---")
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print("\n1. Overall Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Noise', 'Planet', 'EB']))
    
    print("2. Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred))
    
    try:
        roc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
        print(f"\nMacro ROC-AUC: {roc:.4f}")
    except:
        pass

    # Expected Calibration Error (ECE) for Planet class (index 1)
    planet_probs = y_pred_prob[:, 1]
    planet_true = (y_test == 1).astype(int)
    
    brier_score = np.mean((planet_probs - planet_true)**2)
    print(f"Brier Score (Planet): {brier_score:.4f}")
    
    bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        bin_mask = (planet_probs >= bins[i]) & (planet_probs < bins[i+1])
        if np.any(bin_mask):
            bin_acc = np.mean(planet_true[bin_mask])
            bin_conf = np.mean(planet_probs[bin_mask])
            ece += np.sum(bin_mask) / len(planet_probs) * np.abs(bin_acc - bin_conf)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    print("\n--- Diagnostic: Recovery of the 10 Targeted Planets ---")
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
    
    enable_dropout(model)
    recovered = 0
    
    for target in target_tics:
        idx = np.where(tics == target)[0]
        if len(idx) == 0:
            continue
            
        idx = idx[0]
        X_sample = X[idx:idx+1]
        
        means, stds = predict_with_uncertainty(model, X_sample, num_samples=30)
        mean_prob = means[0]
        std_prob = stds[0]
        pred_class = np.argmax(mean_prob)
        
        planet_mean = mean_prob[1]
        planet_std = std_prob[1]
        
        if pred_class == 1:
            recovered += 1
            status = "RECOVERED"
            print(f"{target}: {status} | P(Planet) = {planet_mean:.3f} ± {planet_std:.3f}")
        else:
            status = "MISSED"
            print(f"{target}: {status} | Predicted: {pred_class} | P(Planet) = {planet_mean:.3f} ± {planet_std:.3f}")
            
    print(f"\nTotal Recovered: {recovered}/{len(target_tics)}")

if __name__ == '__main__':
    evaluate_model()
