import os
import uuid
import threading
import re
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Import our new modular core
from src.core.astronomy import get_folded_lightcurve
from src.core.inference import normalize_flux, predict_planet
from src.core.xai import compute_gradcam, compute_integrated_gradients, compute_shap, run_ablation

app = Flask(__name__)

# Load model globally
MODEL_PATH = os.path.join('data', 'models', 'exoplanet_cnn_v2_ternary.keras')
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

BATCH_JOBS = {}

def analyze_star(star_id):
    """Orchestrates the astronomy, inference, and XAI modules."""
    star_id = star_id.strip()
    # Ensure there's only one space between TIC and the number
    star_id = re.sub(r'\s+', ' ', star_id)
    if not star_id.startswith('TIC'):
        star_id = f"TIC {star_id}"
    star_id = star_id.replace("TIC TIC ", "TIC ")
    
    # 1. Astronomy (Fetch & Fold)
    astro_res = get_folded_lightcurve(star_id)
    if not astro_res['success']:
        return {'success': False, 'error': astro_res['error'], 'star_id': star_id}
        
    flux = astro_res['flux']
    
    # 2. Inference (Normalize & Veto Check)
    X_scaled, veto_triggered = normalize_flux(flux)
    
    if veto_triggered:
        prediction = [1.0, 0.0, 0.0]
        uncertainty = [0.0, 0.0, 0.0]
        upsampled_heatmap_conv1 = [0.0] * 2000
        upsampled_heatmap_conv3 = [0.0] * 2000
        heatmap_ig = [0.0] * 2000
        heatmap_shap = [0.0] * 2000
    else:
        # 3. Predict (with MC Dropout Uncertainty) returns arrays of shape (3,)
        prediction, uncertainty = predict_planet(model, X_scaled)
        
        # Target the most likely class for XAI explanation
        target_class = int(np.argmax(prediction))
        
        # 4. XAI Consensus (Grad-CAM, IG, SHAP)
        upsampled_heatmap_conv1 = compute_gradcam(model, X_scaled, 'conv1d', target_class)
        upsampled_heatmap_conv3 = compute_gradcam(model, X_scaled, 'conv1d_2', target_class)
        heatmap_ig = compute_integrated_gradients(model, X_scaled, 50, target_class)
        heatmap_shap = compute_shap(model, X_scaled, target_class)
        
    flux_data = X_scaled[0].flatten().tolist()
    
    return {
        'success': True,
        'prediction': prediction,
        'uncertainty': uncertainty,
        'period': astro_res['period'],
        'duration_hours': astro_res['duration_hours'],
        'depth_ppt': astro_res['depth_ppt'],
        'flux_data': flux_data,
        'heatmap_conv1': upsampled_heatmap_conv1,
        'heatmap_conv3': upsampled_heatmap_conv3,
        'heatmap_ig': heatmap_ig,
        'heatmap_shap': heatmap_shap,
        'star_id': star_id,
        'veto': veto_triggered
    }

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/discovery')
def discovery():
    return render_template('discovery.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    star_id = data.get('star_id', '')
    res = analyze_star(star_id)
    if not res.get('success'):
        return jsonify({'error': res.get('error')})
    return jsonify(res)

@app.route('/ablation', methods=['POST'])
def ablation():
    try:
        data = request.get_json()
        flux_data = data.get('flux_data')
        heatmap = data.get('heatmap')
        original_prediction = data.get('original_prediction')
        target_class = data.get('target_class', 1)
        
        if not flux_data or not heatmap or original_prediction is None:
            return jsonify({'success': False, 'error': 'Missing required data for ablation.'})

        X_base = np.array(flux_data).reshape((1, 2000, 1))
        results = run_ablation(model, X_base, heatmap, target_class, original_prediction)
        
        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/technical')
def technical():
    return render_template('technical.html')

@app.route('/api/full_report', methods=['POST'])
def full_report():
    data = request.get_json()
    star_id = data.get('star_id', '')
    res = analyze_star(star_id)
    
    if not res.get('success'):
        return jsonify({'error': res.get('error')})
        
    if res.get('veto'):
        res['ablation_matrix'] = {}
        res['temperature'] = 1.0
        return jsonify(res)
        
    # Read Calibration Data
    temperature = 1.0
    try:
        import json
        calib_path = os.path.join('data', 'models', 'calibration_params.json')
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                calib = json.load(f)
                temperature = calib.get('temperature', 1.0)
    except Exception as e:
        pass
    res['temperature'] = temperature

    # Run Ablation on ALL 4 XAI Methods simultaneously
    X_base = np.array(res['flux_data']).reshape((1, 2000, 1))
    target_class = int(np.argmax(res['prediction']))
    orig_pred = float(res['prediction'][target_class])
    
    heatmaps = {
        'SHAP': res['heatmap_shap'],
        'Integrated Gradients': res['heatmap_ig'],
        'Grad-CAM (Conv1)': res['heatmap_conv1'],
        'Grad-CAM (Conv3)': res['heatmap_conv3']
    }
    
    ablation_matrix = {}
    for name, hm in heatmaps.items():
        if hm and len(hm) == 2000:
            ablation_matrix[name] = run_ablation(model, X_base, hm, target_class, orig_pred)
            
    res['ablation_matrix'] = ablation_matrix
    return jsonify(res)

def process_batch(job_id, star_ids):
    BATCH_JOBS[job_id]['status'] = 'running'
    for i, star_id in enumerate(star_ids):
        res = analyze_star(star_id)
        BATCH_JOBS[job_id]['results'].append(res)
        BATCH_JOBS[job_id]['progress'] = int(((i + 1) / len(star_ids)) * 100)
    BATCH_JOBS[job_id]['status'] = 'completed'

@app.route('/start_batch', methods=['POST'])
def start_batch():
    data = request.get_json()
    star_ids = data.get('star_ids', [])
    star_ids = [s.strip() for s in star_ids if s.strip()]
    
    job_id = str(uuid.uuid4())
    BATCH_JOBS[job_id] = {'status': 'starting', 'progress': 0, 'results': [], 'total': len(star_ids)}
    
    thread = threading.Thread(target=process_batch, args=(job_id, star_ids))
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/batch_status/<job_id>')
def batch_status(job_id):
    job = BATCH_JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'})
    return jsonify(job)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
