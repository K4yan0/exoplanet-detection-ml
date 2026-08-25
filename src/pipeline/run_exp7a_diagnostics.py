import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                last_conv_layer_name = layer.name
                break
                
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    layer_idx = model.layers.index(last_conv_layer)
    for layer in model.layers[layer_idx + 1:]:
        x = layer(x)
        
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, i] *= pooled_grads[i]
        
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    import scipy.ndimage
    heatmap = scipy.ndimage.zoom(heatmap, img_array.shape[1] / heatmap.shape[0])
    return heatmap

RANDOM_SEED = 42

def main():
    print("--- Exp 7A: Diagnostic Attribution Analysis ---")
    
    path_1sec = 'data/tess_ml_arrays/tess_dataset_exp7_1sec.npz'
    path_5sec = 'data/tess_ml_arrays/tess_dataset_exp7_5sec.npz'
    
    if not os.path.exists(path_1sec) or not os.path.exists(path_5sec):
        print("Error: Datasets not found.")
        return
        
    d1 = np.load(path_1sec)
    d5 = np.load(path_5sec)
    
    if 'tics' not in d1:
        print("Error: 'tics' array not found in dataset. Please run the updated build_exp7_dataset.py first.")
        return
        
    X_1sec, y, tics = d1['X'], d1['y'], d1['tics']
    X_5sec = d5['X']
    
    # Isolate test set exactly as in evaluation
    _, X_test_1sec, _, y_test, _, tics_test = train_test_split(
        X_1sec, y, tics, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    _, X_test_5sec, _, _, _, _ = train_test_split(
        X_5sec, y, tics, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    model_1 = tf.keras.models.load_model('data/models/exp7_1sec_model.keras')
    model_5 = tf.keras.models.load_model('data/models/exp7_5sec_model.keras')
    
    # 1. Identify Planets (y_test == 1)
    planet_indices = np.where(y_test == 1)[0]
    print(f"\nFound {len(planet_indices)} total planets in the test cohort.")
    
    # 2. Get 5-Sector CNN predictions on 5-Sector Input for these planets
    preds_55 = model_5.predict(X_test_5sec[planet_indices], verbose=0)
    classes_55 = np.argmax(preds_55, axis=1)
    
    # 3. Identify the missed planets
    missed_planet_mask = (classes_55 != 1)
    missed_indices = planet_indices[missed_planet_mask]
    
    print(f"\nIdentified {len(missed_indices)} planets missed by the 5-Sector CNN:")
    
    save_dir = 'docs/reports/exp7a_diagnostics'
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in missed_indices:
        tic_id = tics_test[idx]
        print(f" - {tic_id}")
        
        x1 = X_test_1sec[idx:idx+1]
        x5 = X_test_5sec[idx:idx+1]
        
        # Compute Grad-CAM
        cam_11 = make_gradcam_heatmap(x1, model_1)
        cam_55 = make_gradcam_heatmap(x5, model_5)
        
        # Plotting
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # 1-Sector CNN on 1-Sector Input
        axes[0].plot(x1[0, :, 0], color='black', alpha=0.8, linewidth=1, label='1-Sector LC')
        axes[0].scatter(range(2000), x1[0, :, 0], c=cam_11, cmap='jet', s=10, alpha=0.5)
        axes[0].set_title(f"1-Sector CNN (Predicted: {np.argmax(model_1.predict(x1, verbose=0))} | P(Planet): {model_1.predict(x1, verbose=0)[0][1]:.3f})")
        
        # 5-Sector CNN on 5-Sector Input
        axes[1].plot(x5[0, :, 0], color='black', alpha=0.8, linewidth=1, label='5-Sector LC')
        axes[1].scatter(range(2000), x5[0, :, 0], c=cam_55, cmap='jet', s=10, alpha=0.5)
        axes[1].set_title(f"5-Sector CNN (Predicted: {np.argmax(model_5.predict(x5, verbose=0))} | P(Planet): {model_5.predict(x5, verbose=0)[0][1]:.3f})")
        
        plt.suptitle(f"Exp 7A Diagnostic: {tic_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{tic_id}_diagnostic.png"))
        plt.close()
        
    print(f"\nDiagnostics completed. XAI plots saved to {save_dir}")

if __name__ == '__main__':
    main()
