import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

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
    import scipy.ndimage
    heatmap = scipy.ndimage.zoom(heatmap, sequence.shape[1] / len(heatmap))
    return heatmap

def main():
    os.makedirs('docs/assets', exist_ok=True)
    print("Loading models and data...")
    dataset_path = 'data/tess_ml_arrays/tess_dataset_ternary.npz'
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    model_z = load_model('data/models/exp5_reference_model.keras')
    model_m = load_model('data/models/exp6_native_mad_model.keras')
    
    # Get a clear planet
    planet_indices = np.where(Y == 1)[0]
    idx = planet_indices[42]
    x_single_raw = X_raw[idx:idx+1]
    
    x_zscore = get_zscore_data(x_single_raw)
    x_mad = get_mad_data(x_single_raw)
    
    hm_z = compute_gradcam(model_z, x_zscore, 1, 'conv1d_2')
    hm_m = compute_gradcam(model_m, x_mad, 1, 'conv1d_2')
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Native Z-Score
    axes[0].plot(x_zscore[0], color='black', alpha=0.7)
    axes[0].imshow(np.tile(hm_z, (10, 1)), aspect='auto', cmap='jet', alpha=0.3,
                   extent=[0, 2000, np.min(x_zscore), np.max(x_zscore)])
    axes[0].set_title('Native Z-Score CNN Representation (Exp 5)')
    axes[0].set_ylabel('Flux (Z-Score)')
    
    # Native MAD
    axes[1].plot(x_mad[0], color='black', alpha=0.7)
    axes[1].imshow(np.tile(hm_m, (10, 1)), aspect='auto', cmap='jet', alpha=0.3,
                   extent=[0, 2000, np.min(x_mad), np.max(x_mad)])
    axes[1].set_title('Native MAD CNN Representation (Exp 6)')
    axes[1].set_ylabel('Flux (MAD)')
    axes[1].set_xlabel('Phase Bins')
    
    plt.tight_layout()
    save_path = 'docs/assets/exp6_native_xai_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved XAI Comparison to {save_path}")

if __name__ == '__main__':
    main()
