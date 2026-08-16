import numpy as np

def normalize_flux(flux):
    """Applies Z-Score Normalization (Mean & Std) to match train_model.py. No clipping."""
    X_raw = np.array([flux])
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    
    X_scaled = (X_raw - mean) / (std + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    # Check if a massive flare is throwing off the Z-score (basic heuristic veto)
    veto_triggered = bool(np.max(X_scaled) > 10.0)
    
    metadata = {
        'mean': float(mean[0][0]),
        'std': float(std[0][0])
    }
    
    return X_scaled, veto_triggered, metadata

def predict_planet(model, X_scaled, n_iterations=50):
    """Runs the 1D CNN model using Monte Carlo Dropout for uncertainty estimation (Ternary Classifier)."""
    X = X_scaled.reshape((1, 2000, 1))
    
    # Run multiple forward passes with Dropout enabled (training=True)
    predictions = []
    for _ in range(n_iterations):
        pred = model(X, training=True)
        predictions.append(pred[0].numpy())
        
    predictions = np.array(predictions) # Shape: (50, 3)
    mean_predictions = np.mean(predictions, axis=0).tolist()
    uncertainties = np.std(predictions, axis=0).tolist()
    
    return mean_predictions, uncertainties
