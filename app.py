import os
import numpy as np
import pandas as pd
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg') # Prevent GUI threading issues in Flask
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model globally
MODEL_PATH = os.path.join('data', 'models', 'exoplanet_cnn_v1.keras')
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    star_id = data.get('star_id', '').strip()
    
    if not star_id.startswith('TIC'):
        star_id = f"TIC {star_id}"
    star_id = star_id.replace("TIC TIC ", "TIC ")
    
    try:
        search_result = lk.search_lightcurve(star_id, mission='TESS', author='SPOC')
        if len(search_result) == 0:
            return jsonify({'error': f'No SPOC data found for {star_id}. Try a different star!'})
            
        lc = search_result[0].download()
        if lc is None:
            return jsonify({'error': 'Download failed from NASA MAST.'})
            
        flattened_lc = lc.flatten(window_length=101)
        periodogram = flattened_lc.to_periodogram(method='bls', period=np.linspace(1, 20, 100000))
        best_period = periodogram.period_at_max_power
        best_epoch = periodogram.transit_time_at_max_power
        
        folded_lc = flattened_lc.fold(period=best_period, epoch_time=best_epoch)
        binned_lc = folded_lc.bin(bins=2000)
        
        flux = binned_lc.flux.value
        if np.isnan(flux).any():
            flux = pd.Series(flux).interpolate(limit_direction='both').values
            
        if len(flux) != 2000 or np.isnan(flux).any():
            return jsonify({'error': 'Could not extract a clean 2000-point array.'})
            
        X_raw = np.array([flux])
        X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
        mean = np.mean(X_raw, axis=1, keepdims=True)
        std = np.std(X_raw, axis=1, keepdims=True)
        X_scaled = (X_raw - mean) / (std + 1e-8)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        X = X_scaled.reshape((1, 2000, 1))
        prediction = float(model.predict(X, verbose=0)[0][0])
        
        # Plotting
        plt.figure(figsize=(8, 4))
        color = '#00ffcc' if prediction > 0.5 else '#ff3366'
        plt.plot(X_scaled[0].flatten(), color=color, linewidth=1.5)
        
        # Beautiful styling
        plt.gca().set_facecolor('#0f172a')
        plt.gcf().patch.set_facecolor('#0f172a')
        plt.title(f"{star_id} Folded Light Curve", color='white', pad=15)
        plt.xlabel('Phase Bins', color='gray')
        plt.ylabel('Normalized Flux (Z-Score)', color='gray')
        plt.tick_params(colors='gray')
        plt.grid(color='#1e293b', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save image
        os.makedirs('static/plots', exist_ok=True)
        plot_filename = f'{star_id.replace(" ", "_")}.png'
        plot_path = os.path.join('static', 'plots', plot_filename)
        plt.savefig(plot_path, dpi=120)
        plt.close()
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'period': float(best_period.value),
            'image_url': f'/static/plots/{plot_filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
