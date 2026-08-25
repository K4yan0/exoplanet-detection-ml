import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def mad_scaling(x):
    median = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - median), axis=1, keepdims=True)
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-8, mad)
    x_scaled = (x - median) / (mad * 1.4826)
    return x_scaled

def z_score_scaling(x):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.where(std == 0, 1e-8, std)
    return (x - mean) / std

def compute_gradcam(model, sequence, class_index, layer_name):
    # 1. Model that maps image to last conv layer
    last_conv_layer = model.get_layer(layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # 2. Model that maps last conv layer to predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    # Get layers after last_conv_layer
    layer_idx = model.layers.index(last_conv_layer)
    for layer in model.layers[layer_idx + 1:]:
        x = layer(x)
        
    classifier_model = tf.keras.Model(classifier_input, x)

    # 3. Compute gradients
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(sequence)
        tape.watch(last_conv_layer_output)
        
        # Compute predictions
        preds = classifier_model(last_conv_layer_output)
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    heatmap = heatmap.numpy()
    import scipy.ndimage
    heatmap = scipy.ndimage.zoom(heatmap, sequence.shape[1] / len(heatmap))
    return heatmap

def mc_dropout_variance(model, sequence, num_passes=50):
    predictions = []
    # Force training=True to enable dropout during inference
    for _ in range(num_passes):
        pred = model(sequence, training=True)
        predictions.append(pred.numpy()[0])
    
    predictions = np.array(predictions) # Shape: (50, 3)
    variances = np.var(predictions, axis=0) # Variance per class
    mean_variance = np.mean(variances) # Mean variance across all classes
    return mean_variance

def generate_assets():
    os.makedirs('docs/assets', exist_ok=True)
    print("Loading data and model...")
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'tess_dataset_full.npz')
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    model = load_model(os.path.join('data', 'models', 'exoplanet_cnn_v2_ternary.keras'))
    
    # --- ASSET 1: GRAD-CAM COMPARISON ---
    print("Generating Grad-CAM Comparison...")
    # Find a good planet candidate (Y == 1)
    planet_indices = np.where(Y == 1)[0]
    idx = planet_indices[42] # Pick a specific one
    
    x_single_raw = X_raw[idx:idx+1]
    
    x_zscore = z_score_scaling(x_single_raw)
    x_mad = mad_scaling(x_single_raw)
    
    x_zscore = x_zscore.reshape((1, 2000, 1))
    x_mad = x_mad.reshape((1, 2000, 1))
    
    heatmap_zscore = compute_gradcam(model, x_zscore, 1, 'conv1d_2')
    heatmap_mad = compute_gradcam(model, x_mad, 1, 'conv1d_2')
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(x_zscore[0], color='black', alpha=0.7)
    im1 = axes[0].imshow(np.tile(heatmap_zscore, (10, 1)), aspect='auto', cmap='jet', alpha=0.3,
                   extent=[0, 2000, np.min(x_zscore), np.max(x_zscore)])
    axes[0].set_title('Frozen Model on Z-Score Normalized Input (Baseline)')
    axes[0].set_ylabel('Flux (Z-Score)')
    
    axes[1].plot(x_mad[0], color='black', alpha=0.7)
    im2 = axes[1].imshow(np.tile(heatmap_mad, (10, 1)), aspect='auto', cmap='jet', alpha=0.3,
                   extent=[0, 2000, np.min(x_mad), np.max(x_mad)])
    axes[1].set_title('Frozen Model on MAD-Scaled Input (Representation Shift)')
    axes[1].set_ylabel('Flux (MAD)')
    axes[1].set_xlabel('Phase Bins')
    
    plt.tight_layout()
    plt.savefig('docs/assets/mad_gradcam_comparison.png', dpi=300)
    plt.close()
    
    # --- ASSET 2: UNCERTAINTY PLOT ---
    print("Generating Predictive Uncertainty Plot (MC-Dropout)...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_raw), 200, replace=False)
    
    variances_zscore = []
    variances_mad = []
    
    for i, s_idx in enumerate(sample_indices):
        x_raw = X_raw[s_idx:s_idx+1]
        
        x_z = z_score_scaling(x_raw).reshape((1, 2000, 1))
        x_m = mad_scaling(x_raw).reshape((1, 2000, 1))
        
        v_z = mc_dropout_variance(model, x_z)
        v_m = mc_dropout_variance(model, x_m)
        
        variances_zscore.append(v_z)
        variances_mad.append(v_m)
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/200 samples for variance...")
            
    plt.figure(figsize=(8, 6))
    sns.kdeplot(variances_zscore, fill=True, label='Z-Score (Baseline)', color='blue', alpha=0.5)
    sns.kdeplot(variances_mad, fill=True, label='MAD Scaling', color='red', alpha=0.5)
    plt.title('MC-Dropout Predictive Uncertainty: Z-Score vs MAD Scaling\n(Frozen CNN Weights)')
    plt.xlabel('Mean Variance (Epistemic Uncertainty)')
    plt.ylabel('Density of Predictions')
    plt.axvline(np.mean(variances_zscore), color='blue', linestyle='--', label=f'Z-Score Mean: {np.mean(variances_zscore):.4f}')
    plt.axvline(np.mean(variances_mad), color='red', linestyle='--', label=f'MAD Mean: {np.mean(variances_mad):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/assets/mad_uncertainty_plot.png', dpi=300)
    plt.close()
    
    print("Done! Check docs/assets/")

if __name__ == '__main__':
    generate_assets()
