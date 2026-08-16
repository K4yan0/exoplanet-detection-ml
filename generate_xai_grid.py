import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.core.astronomy import get_folded_lightcurve
from src.core.inference import normalize_flux
from src.core.xai import compute_gradcam, compute_integrated_gradients, compute_shap

def main():
    print("Loading model...")
    model = load_model(os.path.join('data', 'models', 'exoplanet_cnn_v2_ternary.keras'))
    
    star_id = "TIC 185259483"
    print(f"Fetching data for {star_id}...")
    astro_res = get_folded_lightcurve(star_id)
    if not astro_res['success']:
        print(f"Error fetching data: {astro_res['error']}")
        return
        
    print("Normalizing...")
    X_scaled, veto, _ = normalize_flux(astro_res['flux'])
    flux_data = X_scaled[0].flatten()
    
    target_class = 2 # Eclipsing Binary
    
    print("Computing XAI...")
    heatmaps = {
        'Grad-CAM (Conv1)': compute_gradcam(model, X_scaled, 'conv1d', target_class),
        'Grad-CAM (Conv3)': compute_gradcam(model, X_scaled, 'conv1d_2', target_class),
        'Integrated Gradients': compute_integrated_gradients(model, X_scaled, 50, target_class),
        'SHAP': compute_shap(model, X_scaled, target_class)
    }
    
    print("Plotting...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'XAI Consensus - {star_id} (Eclipsing Binary)', fontsize=16, color='white')
    
    phases = np.linspace(-0.5, 0.5, 2000)
    
    for ax, (title, heatmap) in zip(axes.flatten(), heatmaps.items()):
        # Plot the base light curve
        ax.plot(phases, flux_data, color='white', alpha=0.3, linewidth=1)
        
        # Normalize heatmap for coloring
        h_norm = np.array(heatmap)
        if np.max(h_norm) > 0:
            h_norm = h_norm / np.max(h_norm)
            
        # Scatter with heatmap colors
        scatter = ax.scatter(phases, flux_data, c=h_norm, cmap='cool', s=10, alpha=0.8)
        
        ax.set_title(title, color='cyan')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux')
        ax.grid(True, alpha=0.1)
        
    plt.tight_layout()
    
    save_path = os.path.join('docs', 'assets', 'xai_grid_tic185259483.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    print(f"Saved real data plot to {save_path}")

if __name__ == '__main__':
    main()
