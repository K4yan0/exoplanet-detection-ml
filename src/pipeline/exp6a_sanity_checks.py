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

def compute_gradcam_raw(model, sequence, class_index, layer_name):
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
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap.numpy()
    
    if np.isnan(heatmap).any() or np.max(heatmap) == 0:
        return np.zeros(2000)
        
    import scipy.ndimage
    heatmap = scipy.ndimage.zoom(heatmap, sequence.shape[1] / len(heatmap))
    return heatmap

def main():
    print("Loading models and data for Exp 6A Sanity Checks...")
    dataset_path = 'data/tess_ml_arrays/tess_dataset_ternary.npz'
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    _, X_test, _, y_test = train_test_split(X_raw, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
    
    planet_indices = np.where(y_test == 1)[0]
    X_planets = X_test[planet_indices]
    
    model_z = tf.keras.models.load_model('data/models/exp5_reference_model.keras')
    model_m = tf.keras.models.load_model('data/models/exp6_native_mad_model.keras')
    
    paired_correlations = []
    
    all_z_maps = []
    all_m_maps = []
    
    for i in range(len(X_planets)):
        x_single = X_planets[i:i+1]
        
        x_z = get_zscore_data(x_single)
        x_m = get_mad_data(x_single)
        
        hm_z = compute_gradcam_raw(model_z, x_z, 1, 'conv1d_2')
        hm_m = compute_gradcam_raw(model_m, x_m, 1, 'conv1d_2')
        
        if np.max(hm_z) == 0 or np.max(hm_m) == 0:
            continue
            
        hm_z_norm = (hm_z - np.min(hm_z)) / (np.max(hm_z) - np.min(hm_z))
        hm_m_norm = (hm_m - np.min(hm_m)) / (np.max(hm_m) - np.min(hm_m))
        
        all_z_maps.append(hm_z_norm)
        all_m_maps.append(hm_m_norm)
        
        corr, _ = pearsonr(hm_z_norm, hm_m_norm)
        paired_correlations.append(corr if not np.isnan(corr) else 0)
        
    observed_mean_corr = np.mean(paired_correlations)
    
    # 10,000 Permutations for exact p-value
    print("\nRunning 10,000 permutations for null distribution...")
    null_mean_corrs = []
    np.random.seed(42)
    
    for _ in range(10000):
        shuffled_m = all_m_maps.copy()
        np.random.shuffle(shuffled_m)
        
        iteration_corrs = []
        for i in range(len(all_z_maps)):
            c, _ = pearsonr(all_z_maps[i], shuffled_m[i])
            iteration_corrs.append(c if not np.isnan(c) else 0)
            
        null_mean_corrs.append(np.mean(iteration_corrs))
    
    null_mean_corrs = np.array(null_mean_corrs)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(null_mean_corrs) >= np.abs(observed_mean_corr))

    print("==================================================")
    print("EXP 6A: STATISTICAL XAI VALIDATION")
    print("==================================================")
    
    print("\n1. Paired Correlation (Same Target):")
    print(f"   Mean Pearson r: {observed_mean_corr:.4f}")
    
    print("\n2. Null Distribution (10,000 Random Shuffles):")
    print(f"   Expected Null Mean: {np.mean(null_mean_corrs):.4f}")
    print(f"   95% Confidence Interval for Null Mean: [{np.percentile(null_mean_corrs, 2.5):.4f}, {np.percentile(null_mean_corrs, 97.5):.4f}]")
    
    print("\n3. Permutation Test:")
    print(f"   Observed difference from Expected Null: {(observed_mean_corr - np.mean(null_mean_corrs)):.4f}")
    print(f"   p-value: {p_value:.4f}")
    
    print("\nStatistical Verdict:")
    if p_value < 0.05:
        print("   The paired correlation is significantly different from random noise.")
    else:
        print("   FAIL TO REJECT NULL HYPOTHESIS.")
        print("   Under this Grad-CAM comparison, the Native Z-Score and Native MAD models")
        print("   showed no detectable increase in attribution-map correlation for the same")
        print("   planetary targets relative to randomly paired targets.")

if __name__ == '__main__':
    main()
