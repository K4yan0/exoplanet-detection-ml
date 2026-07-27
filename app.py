import os
import numpy as np
import pandas as pd
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras.models import load_model

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
        median = np.median(X_raw, axis=1, keepdims=True)
        mad = np.median(np.abs(X_raw - median), axis=1, keepdims=True)
        mad_scaled = mad * 1.4826
        
        X_scaled = (X_raw - median) / (mad_scaled + 1e-8)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # One-Sided Clip: Cap positive flares at +3.0
        X_scaled = np.clip(X_scaled, a_min=None, a_max=3.0)
        
        # HEURISTIC VETO
        num_clipped_points = int(np.sum(X_scaled == 3.0))
        veto_triggered = bool(num_clipped_points > 50)
        
        if veto_triggered:
            prediction = 0.0
        else:
            X = X_scaled.reshape((1, 2000, 1))
            prediction = float(model.predict(X, verbose=0)[0][0])
        
        # Return data for interactive plotting on the frontend
        flux_data = X_scaled[0].flatten().tolist()
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'period': float(best_period.value),
            'flux_data': flux_data,
            'star_id': star_id,
            'veto': veto_triggered
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
