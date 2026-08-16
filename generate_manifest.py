import json
import hashlib
import sys
import subprocess
import os

def generate_manifest():
    model_path = os.path.join('data', 'models', 'exoplanet_cnn_v2_ternary.keras')
    with open(model_path, 'rb') as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
        
    try:
        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8').split('\n')
    except:
        reqs = []

    manifest = {
        "version": "v1.0.0",
        "model": {
            "name": "exoplanet_cnn_v2_ternary.keras",
            "sha256_hash": model_hash,
            "architecture": "1D CNN Ternary Classifier (Softmax)"
        },
        "dataset": {
            "name": "tess_dataset_ternary.npz",
            "sources": ["tess_positive_targets.json", "tess_eb_targets.json"],
            "test_set_split": "train_test_split(test_size=0.2, random_state=42, stratify=Y)"
        },
        "preprocessing_contract": {
            "version": "TERNARY_V1",
            "sectors_used": 1,
            "outlier_removal": False,
            "filter": "Savitzky-Golay",
            "window_length": 101,
            "phase_folded": True,
            "num_bins": 2000,
            "scaling": "Z-Score (Mean/Std)",
            "clipping": False
        },
        "training_configuration": {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "random_seed": 42
        },
        "xai_configuration": {
            "methods": ["Grad-CAM", "Integrated Gradients", "SHAP"],
            "gradcam_layers": ["conv1d", "conv1d_2"],
            "ig_steps": 50,
            "baseline_artifact": "docs/assets/xai_grid_tic185259483.png"
        },
        "environment": {
            "python_version": sys.version,
            "dependencies": [r for r in reqs if r]
        }
    }
    
    os.makedirs('docs', exist_ok=True)
    with open('docs/v1_reproducibility_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print("Manifest created successfully.")

if __name__ == "__main__":
    generate_manifest()
