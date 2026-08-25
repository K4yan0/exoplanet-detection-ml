import numpy as np
import tensorflow as tf
from scipy.signal import correlate

def main():
    data = np.load('data/tess_ml_arrays/tess_dataset_exp8.npz')
    X = data['X']  
    Y = data['y']  
    TICS = data['tics']
    
    model = tf.keras.models.load_model('data/models/exp8_model.keras')
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    
    tps = np.where((Y == 1) & (y_pred == 1))[0]
    fps = np.where((Y == 0) & (y_pred == 1))[0]
    
    def compute_cc_lags(idx):
        x_sample = X[idx] # (5, 2000, 1)
        lags_out = []
        base_flux = x_sample[0, :, 0]
        
        # If the base sector is flat, cross-correlation is meaningless
        if np.std(base_flux) < 0.1:
            return [np.nan]*4
            
        for i in range(1, 5):
            comp_flux = x_sample[i, :, 0]
            if np.std(comp_flux) < 0.1:
                lags_out.append(np.nan)
                continue
                
            # Cross-correlate
            cc = correlate(comp_flux - np.mean(comp_flux), base_flux - np.mean(base_flux), mode='same')
            best_lag_idx = np.argmax(cc)
            
            # center is 1000. So best_lag = best_lag_idx - 1000
            best_lag = best_lag_idx - 1000
            
            # Phase drift
            phase_drift = best_lag / 2000.0
            lags_out.append(phase_drift)
            
        return lags_out

    print("--- Phase Drifts (Cross-Correlation) for True Planets ---")
    for i in range(min(5, len(tps))):
        idx = tps[i]
        lags = compute_cc_lags(idx)
        print(f"{TICS[idx]}: {['{:.3f}'.format(x) if not np.isnan(x) else 'NaN' for x in lags]}")

    print("\n--- Phase Drifts (Cross-Correlation) for False Positives ---")
    for i in range(min(5, len(fps))):
        idx = fps[i]
        lags = compute_cc_lags(idx)
        print(f"{TICS[idx]}: {['{:.3f}'.format(x) if not np.isnan(x) else 'NaN' for x in lags]}")

if __name__ == '__main__':
    main()
