import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

def measure_transit(lc, window=(900, 1100)):
    """
    Measure the depth and width of the central transit in the folded light curve.
    Assumes light curves are Z-scored, so the transit is negative.
    """
    segment = lc[window[0]:window[1]]
    depth = abs(np.min(segment))
    
    # Calculate Full Width at Half Maximum (FWHM)
    half_max = -depth / 2.0
    below_half = np.where(segment < half_max)[0]
    if len(below_half) > 0:
        width = below_half[-1] - below_half[0]
    else:
        width = 0
    return depth, width

def main():
    print("--- Exp 7A: Quantifying Morphological Smearing ---")
    path_1sec = 'data/tess_ml_arrays/tess_dataset_exp7_1sec.npz'
    path_5sec = 'data/tess_ml_arrays/tess_dataset_exp7_5sec.npz'
    model_path = 'data/models/exp7_5sec_model.keras'

    if not os.path.exists(path_1sec) or not os.path.exists(path_5sec) or not os.path.exists(model_path):
        print("Required datasets or model not found.")
        return

    # Load data
    d1 = np.load(path_1sec)
    d5 = np.load(path_5sec)
    X_1sec, y, tics = d1['X'], d1['y'], d1['tics']
    X_5sec = d5['X']

    # Get the test cohort (using same random seed to align with diagnostic script)
    _, X_test_1, _, y_test, _, tics_test = train_test_split(X_1sec, y, tics, test_size=0.2, random_state=42, stratify=y)
    _, X_test_5, _, _, _, _ = train_test_split(X_5sec, y, tics, test_size=0.2, random_state=42, stratify=y)

    # Load 5-sector model
    print("Loading 5-Sector CNN...")
    model_5 = tf.keras.models.load_model(model_path)
    
    # Identify missed planets
    planet_indices = np.where(y_test == 1)[0]
    preds_55 = model_5.predict(X_test_5[planet_indices], verbose=0)
    classes_55 = np.argmax(preds_55, axis=1)
    missed_indices = planet_indices[classes_55 != 1]
    
    print(f"\nFound {len(missed_indices)} missed planets to analyze.")

    depth_changes = []
    width_changes = []

    for idx in missed_indices:
        d1, w1 = measure_transit(X_test_1[idx, :, 0])
        d5, w5 = measure_transit(X_test_5[idx, :, 0])
        
        if d1 > 0 and d5 > 0:
            depth_changes.append((d5 - d1) / d1 * 100)
        if w1 > 0 and w5 > 0:
            width_changes.append((w5 - w1) / w1 * 100)
            
    print("\n--- Quantification Results ---")
    print(f"Median Depth Change (1-Sector -> 5-Sector): {np.median(depth_changes):.2f}%")
    print(f"Median Width Change (1-Sector -> 5-Sector): {np.median(width_changes):.2f}%")

if __name__ == "__main__":
    main()
