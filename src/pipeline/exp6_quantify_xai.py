import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

RANDOM_SEED = 42

def get_zscore_data(X):
    epsilon = 1e-8
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    X_norm = (X - mean) / (std + epsilon)
    return np.expand_dims(X_norm, axis=2)

def get_mad_data(X):
    median = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - median), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1e-8, mad)
    X_norm = (X - median) / (mad * 1.4826)
    return np.expand_dims(X_norm, axis=2)

def compute_gradcam(model, sequence, class_index, layer_name):
    last_conv_layer = model.get_layer(layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    layer_idx = model.layers.index(last_conv_layer)
    for layer in model.layers[layer_idx + 1:]:
        x = layer(x)
        
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(sequence)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    heatmap = heatmap.numpy()
    if np.isnan(heatmap).any() or np.sum(heatmap) == 0:
        heatmap = np.zeros_like(heatmap)
    else:
        import scipy.ndimage
        heatmap = scipy.ndimage.zoom(heatmap, sequence.shape[1] / len(heatmap))
    
    # Normalize to sum to 1 so it represents an energy distribution
    total_energy = np.sum(heatmap)
    if total_energy > 0:
        heatmap = heatmap / total_energy
        
    return heatmap

def main():
    print("Loading models and data...")
    dataset_path = 'data/tess_ml_arrays/tess_dataset_ternary.npz'
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Test set
    _, X_test, _, y_test = train_test_split(X_raw, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
    
    # Isolate Planets
    planet_indices = np.where(y_test == 1)[0]
    X_planets = X_test[planet_indices]
    
    print(f"Quantifying XAI representations for {len(X_planets)} planetary transits...")
    
    model_z = tf.keras.models.load_model('data/models/exp5_reference_model.keras')
    model_m = tf.keras.models.load_model('data/models/exp6_native_mad_model.keras')
    
    mses = []
    correlations = []
    
    # Center transit window is typically around phase 0.5 (bins 900 to 1100 out of 2000)
    z_energy_in_transit = []
    m_energy_in_transit = []
    
    for i in range(len(X_planets)):
        x_single_raw = X_planets[i:i+1]
        
        x_z = get_zscore_data(x_single_raw)
        x_m = get_mad_data(x_single_raw)
        
        hm_z = compute_gradcam(model_z, x_z, 1, 'conv1d_2')
        hm_m = compute_gradcam(model_m, x_m, 1, 'conv1d_2')
        
        if np.sum(hm_z) == 0 or np.sum(hm_m) == 0:
            continue # Skip flat heatmaps
            
        mse = np.mean((hm_z - hm_m)**2)
        corr, _ = pearsonr(hm_z, hm_m)
        
        mses.append(mse)
        correlations.append(corr if not np.isnan(corr) else 0)
        
        # Energy inside transit (bins 900-1100) vs outside
        # Heatmap sums to 1.0
        z_transit = np.sum(hm_z[850:1150])
        m_transit = np.sum(hm_m[850:1150])
        
        z_energy_in_transit.append(z_transit)
        m_energy_in_transit.append(m_transit)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(X_planets)} targets...")
            
    print("\n==================================================")
    print("XAI REPRESENTATION QUANTIFICATION (Native Z vs Native MAD)")
    print("==================================================")
    
    print(f"\n1. Mean Squared Error (MSE) between heatmaps: {np.mean(mses):.6f}")
    print(f"2. Mean Pearson Correlation between heatmaps: {np.mean(correlations):.4f}")
    
    print("\n3. Attribution Focus (Energy inside central transit [bins 850-1150]):")
    print(f"   Native Z-Score CNN: {np.mean(z_energy_in_transit)*100:.2f}% of attention inside transit")
    print(f"   Native MAD CNN:     {np.mean(m_energy_in_transit)*100:.2f}% of attention inside transit")
    
    print("\nInterpretation:")
    if np.mean(correlations) > 0.8:
        print("-> HIGH CORRELATION: The models learned nearly identical representations.")
    elif np.mean(correlations) > 0.5:
        print("-> MODERATE CORRELATION: The models learned similar, but quantitatively distinct representations.")
    else:
        print("-> LOW CORRELATION: The models formed fundamentally different neural representations of the identical targets.")

if __name__ == '__main__':
    main()
