import numpy as np
import tensorflow as tf

def main():
    # Load dataset
    data = np.load('data/tess_ml_arrays/tess_dataset_exp8.npz')
    X = data['X']  # Shape (248, 5, 2000, 1)
    Y = data['y']  # Shape (248,)
    TICS = data['tics']
    
    # Load Exp 8 model to find False Positives
    model = tf.keras.models.load_model('data/models/exp8_model.keras')
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Find some True Positives (Planet correctly classified)
    tps = np.where((Y == 1) & (y_pred == 1))[0]
    # Find False Positives (Noise incorrectly classified as Planet)
    fps = np.where((Y == 0) & (y_pred == 1))[0]
    
    def compute_phase_offsets(idx):
        x_sample = X[idx] # (5, 2000, 1)
        offsets = []
        for i in range(5):
            sector_flux = x_sample[i, :, 0]
            # Check if sector is completely flat (all zeros or very low variance)
            if np.std(sector_flux) < 0.1:
                offsets.append(np.nan)
                continue
                
            # The deepest point of the transit
            deepest_bin = np.argmin(sector_flux)
            
            # Phase is from -0.5 to 0.5, where center (bin 1000) is 0.0
            phase_offset = (deepest_bin / 2000.0) - 0.5
            offsets.append(phase_offset)
        return offsets
        
    print("--- Phase Offsets for True Planets (Consistent Timing Expected) ---")
    for i in range(min(5, len(tps))):
        idx = tps[i]
        offsets = compute_phase_offsets(idx)
        print(f"{TICS[idx]} [Label: {Y[idx]}, Pred: {y_pred[idx]}]: {['{:.3f}'.format(x) if not np.isnan(x) else 'NaN' for x in offsets]}")

    print("\n--- Phase Offsets for False Positives (Inconsistent Timing Expected) ---")
    for i in range(min(5, len(fps))):
        idx = fps[i]
        offsets = compute_phase_offsets(idx)
        print(f"{TICS[idx]} [Label: {Y[idx]}, Pred: {y_pred[idx]}]: {['{:.3f}'.format(x) if not np.isnan(x) else 'NaN' for x in offsets]}")

if __name__ == '__main__':
    main()
