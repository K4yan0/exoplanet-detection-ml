import os
import json
import numpy as np
import pandas as pd
import lightkurve as lk
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import socket
import time

socket.setdefaulttimeout(10.0)
original_request = requests.Session.request
def timeout_request(self, method, url, **kwargs):
    if 'timeout' not in kwargs or kwargs['timeout'] is None:
        kwargs['timeout'] = 10
    return original_request(self, method, url, **kwargs)
requests.Session.request = timeout_request

def safe_search_and_download(clean_star_id):
    attempt = 1
    while attempt <= 2:
        try:
            search_result = lk.search_lightcurve(clean_star_id, mission='TESS', author='SPOC')
            if len(search_result) == 0:
                return None
            return search_result[0].download()
        except Exception as e:
            time.sleep(1)
            attempt += 1
    return None

def remove_outliers_nondestructive(lc, sigma_upper=3.0, sigma_lower=10.0):
    clean_lc = lc.copy()
    flux = clean_lc.flux.value
    median = np.nanmedian(flux)
    std = np.nanstd(flux)
    outlier_mask = (flux > median + sigma_upper * std) | (flux < median - sigma_lower * std)
    if outlier_mask.any():
        flux[outlier_mask] = np.nan
        flux = pd.Series(flux).interpolate(limit_direction='both').values
    clean_lc.flux = flux
    return clean_lc

def generate_sample(flattened_lc, period, epoch, num_bins=2000):
    folded_lc = flattened_lc.fold(period=period, epoch_time=epoch)
    binned_lc = folded_lc.bin(bins=num_bins)
    flux = binned_lc.flux.value
    if np.isnan(flux).any():
        flux = pd.Series(flux).interpolate(limit_direction='both').values
    if len(flux) != num_bins or np.isnan(flux).any():
        return None
    return flux

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    # 1. Model that maps image to last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                last_conv_layer_name = layer.name
                break
                
    last_conv_layer = model.get_layer(last_conv_layer_name)
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
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        
        # Compute predictions
        preds = classifier_model(last_conv_layer_output)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def find_representative(target_dict, true_label, model_v1, model_exp4, label_name):
    print(f"Searching for representative {label_name} (True={true_label})...")
    for star_id, params in target_dict.items():
        clean_star_id = star_id.replace("TIC TIC ", "TIC ")
        print(f"  Checking {clean_star_id}...", flush=True)
        lc = safe_search_and_download(clean_star_id)
        if lc is None: 
            print(f"    -> Download failed or no data.", flush=True)
            continue
        
        flattened_lc = lc.flatten(window_length=101)
        
        true_period = params['period']
        true_epoch = params['epoch']
        
        # If noise, mangle the period
        if true_label == 0:
            period = true_period * 1.345
            epoch = true_epoch + (true_period * 0.5)
        else:
            period = true_period
            epoch = true_epoch
            
        # V1 Processing
        flux_v1 = generate_sample(flattened_lc, period, epoch)
        if flux_v1 is None: continue
        
        # Exp 4 Processing
        clean_lc = remove_outliers_nondestructive(flattened_lc)
        flux_exp4 = generate_sample(clean_lc, period, epoch)
        if flux_exp4 is None: continue
        # Normalize
        flux_v1_scaled = (flux_v1 - np.mean(flux_v1)) / (np.std(flux_v1) + 1e-8)
        flux_v1_scaled = np.nan_to_num(flux_v1_scaled, nan=0.0)
        
        flux_exp4_scaled = (flux_exp4 - np.mean(flux_exp4)) / (np.std(flux_exp4) + 1e-8)
        flux_exp4_scaled = np.nan_to_num(flux_exp4_scaled, nan=0.0)
        
        X_v1 = flux_v1_scaled.reshape(1, 2000, 1)
        X_exp4 = flux_exp4_scaled.reshape(1, 2000, 1)
        
        # Predict
        pred_v1 = np.argmax(model_v1.predict(X_v1, verbose=0))
        pred_exp4 = np.argmax(model_exp4.predict(X_exp4, verbose=0))
        
        if pred_v1 == true_label and pred_exp4 != true_label:
            print(f"  -> Found! {clean_star_id} | V1 Pred: {pred_v1}, Exp4 Pred: {pred_exp4}")
            return clean_star_id, flattened_lc, clean_lc, flux_v1, flux_exp4, X_v1, X_exp4, pred_v1, pred_exp4
            
    print(f"  -> Could not find a clean mismatch for {label_name}.")
    return None

def plot_xai(star_id, label_name, raw_lc, clean_lc, flux_v1, flux_exp4, heatmap_v1, heatmap_exp4, pred_v1, pred_exp4):
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f"Exp 4A Attribution Analysis: {star_id} ({label_name})", fontsize=16)
    
    classes = {0: "Noise", 1: "Planet", 2: "EB"}
    
    # 1. Raw vs Cleaned LC
    axes[0, 0].plot(raw_lc.time.value, raw_lc.flux.value, 'k.', markersize=2)
    axes[0, 0].set_title("V1: Raw Flattened LC")
    
    axes[0, 1].plot(clean_lc.time.value, clean_lc.flux.value, 'k.', markersize=2)
    axes[0, 1].set_title("Exp 4: Outlier-Removed LC")
    
    # 2. Binned Flux
    axes[1, 0].plot(flux_v1, 'b-')
    axes[1, 0].set_title("V1: Phase-Folded & Binned")
    
    axes[1, 1].plot(flux_exp4, 'b-')
    axes[1, 1].set_title("Exp 4: Phase-Folded & Binned")
    
    # 3. Overlay (Zoomed)
    axes[2, 0].plot(flux_v1, 'b-', label='V1')
    axes[2, 0].plot(flux_exp4, 'r--', label='Exp 4', alpha=0.7)
    axes[2, 0].set_xlim(800, 1200)
    axes[2, 0].set_title("Central Transit Overlay (Zoomed)")
    axes[2, 0].legend()
    
    axes[2, 1].axis('off')
    
    # 4. Grad-CAM
    x_cam = np.linspace(0, 2000, len(heatmap_v1))
    
    axes[3, 0].plot(flux_v1, 'gray', alpha=0.5)
    ax30_tw = axes[3, 0].twinx()
    ax30_tw.plot(x_cam, heatmap_v1, 'g-', linewidth=2)
    axes[3, 0].set_title(f"V1 Grad-CAM | Pred: {classes[pred_v1]}")
    
    axes[3, 1].plot(flux_exp4, 'gray', alpha=0.5)
    ax31_tw = axes[3, 1].twinx()
    ax31_tw.plot(x_cam, heatmap_exp4, 'r-', linewidth=2)
    axes[3, 1].set_title(f"Exp 4 Grad-CAM | Pred: {classes[pred_exp4]}")
    
    plt.tight_layout()
    save_path = os.path.join('docs', f'exp4a_xai_{label_name}.png')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

def main():
    print("Loading models...")
    model_v1 = tf.keras.models.load_model('data/models/aligned_v1.keras')
    model_exp4 = tf.keras.models.load_model('data/models/aligned_exp4.keras')
    
    with open('data/tess_positive_targets.json', 'r') as f:
        targets_pos = json.load(f)
    with open('data/tess_eb_targets.json', 'r') as f:
        targets_eb = json.load(f)
        
    os.makedirs('docs', exist_ok=True)
    
    # Find EB
    res_eb = find_representative(targets_eb, 2, model_v1, model_exp4, "EB")
    if res_eb:
        clean_star_id, raw_lc, clean_lc, flux_v1, flux_exp4, X_v1, X_exp4, p1, p2 = res_eb
        hm_v1 = make_gradcam_heatmap(X_v1, model_v1)
        hm_exp4 = make_gradcam_heatmap(X_exp4, model_exp4)
        plot_xai(clean_star_id, "EB", raw_lc, clean_lc, flux_v1, flux_exp4, hm_v1, hm_exp4, p1, p2)
        
    # Find Planet
    res_pl = find_representative(targets_pos, 1, model_v1, model_exp4, "Planet")
    if res_pl:
        clean_star_id, raw_lc, clean_lc, flux_v1, flux_exp4, X_v1, X_exp4, p1, p2 = res_pl
        hm_v1 = make_gradcam_heatmap(X_v1, model_v1)
        hm_exp4 = make_gradcam_heatmap(X_exp4, model_exp4)
        plot_xai(clean_star_id, "Planet", raw_lc, clean_lc, flux_v1, flux_exp4, hm_v1, hm_exp4, p1, p2)

    # Find Noise
    res_ns = find_representative(targets_pos, 0, model_v1, model_exp4, "Noise")
    if res_ns:
        clean_star_id, raw_lc, clean_lc, flux_v1, flux_exp4, X_v1, X_exp4, p1, p2 = res_ns
        hm_v1 = make_gradcam_heatmap(X_v1, model_v1)
        hm_exp4 = make_gradcam_heatmap(X_exp4, model_exp4)
        plot_xai(clean_star_id, "Noise", raw_lc, clean_lc, flux_v1, flux_exp4, hm_v1, hm_exp4, p1, p2)

if __name__ == '__main__':
    main()
