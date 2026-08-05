import numpy as np
import tensorflow as tf

def compute_gradcam(model, X_scaled, layer_name, prediction):
    """Computes 1D Grad-CAM for a given layer."""
    X = X_scaled.reshape((1, 2000, 1))
    feature_extractor = tf.keras.Model(model.inputs, model.get_layer(layer_name).output)
    
    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor({"input_layer": X})
        tape.watch(conv_outputs)
        
        x_layer = conv_outputs
        layer_names = [layer.name for layer in model.layers]
        start_idx = layer_names.index(layer_name) + 1
        for layer in model.layers[start_idx:]:
            x_layer = layer(x_layer)
        preds = x_layer
        loss = preds[:, 0] if prediction > 0.5 else 1.0 - preds[:, 0]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat > 0:
        heatmap /= max_heat
        
    heatmap = heatmap.numpy()
    original_x = np.linspace(0, 1, 2000)
    heatmap_x = np.linspace(0, 1, len(heatmap))
    upsampled_heatmap = np.interp(original_x, heatmap_x, heatmap).tolist()
    
    return upsampled_heatmap

import shap

def compute_integrated_gradients(model, X_scaled, m_steps=50):
    """Computes Integrated Gradients attribution."""
    X = tf.cast(X_scaled.reshape((1, 2000, 1)), tf.float32)
    # Baseline is a flat light curve (0.0 due to Z-score normalization)
    baseline = tf.zeros_like(X)
    
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    
    # Shape: (m_steps+1, 1, 2000, 1) -> reshape to (m_steps+1, 2000, 1)
    interpolated_images = baseline + alphas_x * (X - baseline)
    interpolated_images = tf.reshape(interpolated_images, (m_steps+1, 2000, 1))
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        preds = model(interpolated_images)
        probs = preds[:, 0]
        
    grads = tape.gradient(probs, interpolated_images)
    avg_grads = tf.reduce_mean(grads[:-1], axis=0) # Shape: (2000, 1)
    
    # Multiply by (inputs - baseline)
    integrated_gradients = tf.squeeze(X - baseline) * tf.squeeze(avg_grads)
    
    # Use absolute attributions for the heatmap
    heatmap = tf.abs(integrated_gradients)
    
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat > 0:
        heatmap /= max_heat
        
    return heatmap.numpy().tolist()

def compute_shap(model, X_scaled):
    """Computes SHAP values using GradientExplainer."""
    X = X_scaled.reshape((1, 2000, 1))
    # Background: flat light curves. We provide a small batch of baseline zeroes.
    background = np.zeros((10, 2000, 1)) 
    
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X)
    
    # Extract the values for the positive class
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
        
    heatmap = np.abs(np.squeeze(shap_vals))
    max_heat = np.max(heatmap)
    if max_heat > 0:
        heatmap /= max_heat
        
    return heatmap.tolist()

def run_ablation(model, X_base, heatmap, original_prediction):
    """Performs perturbation analysis on specific light curve regions."""
    results = []
    
    def run_mask(name, indices):
        X_masked = X_base.copy()
        X_masked[0, indices, 0] = 0.0
        new_pred = float(model.predict(X_masked, verbose=0)[0][0])
        confidence_drop = original_prediction - new_pred
        results.append({
            'name': name,
            'new_prediction': new_pred,
            'confidence_drop': confidence_drop
        })

    # 1. Mask Transit (Centered at 1000, mask 900 to 1100)
    run_mask('Transit Region (Physics)', np.arange(900, 1100))
    
    # 2. Mask Highlighted (Top 30% hottest XAI points)
    threshold = np.percentile(heatmap, 70)
    highlighted_indices = np.where(np.array(heatmap) >= threshold)[0]
    run_mask('XAI Highlighted Region', highlighted_indices)
    
    # 3. Mask Pre-Transit (Baseline check, 300 to 500)
    run_mask('Pre-Transit (Baseline)', np.arange(300, 500))
    
    # 4. Mask Random Region (200 points)
    random_start = np.random.randint(0, 700)
    run_mask('Random Background', np.arange(random_start, random_start + 200))

    return results
