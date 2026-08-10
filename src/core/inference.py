import numpy as np

def normalize_flux(flux):
    """Applies Median Absolute Deviation (MAD) scaling and One-Sided Clipping."""
    X_raw = np.array([flux])
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    median = np.median(X_raw, axis=1, keepdims=True)
    mad = np.median(np.abs(X_raw - median), axis=1, keepdims=True)
    mad_scaled = mad * 1.4826
    
    X_scaled = (X_raw - median) / (mad_scaled + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    # Clip positive flares
    X_scaled = np.clip(X_scaled, a_min=None, a_max=3.0)
    
    num_clipped_points = int(np.sum(X_scaled == 3.0))
    veto_triggered = bool(num_clipped_points > 50)
    
    return X_scaled, veto_triggered

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
