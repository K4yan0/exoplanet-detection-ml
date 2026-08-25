import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt

class ResearchEvaluator:
    """
    Centralized evaluation protocol strictly enforcing the RESEARCH_EVALUATION_PROTOCOL.md.
    Every experiment in Phase III, IV, and V must pass through this class.
    """
    def __init__(self, experiment_name, model_path, X_test, y_test, class_names=None):
        self.experiment_name = experiment_name
        self.model_path = model_path
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names if class_names else ['Noise (0)', 'Planet (1)', 'EB (2)']
        self.n_classes = len(self.class_names)
        
        print(f"[{self.experiment_name}] Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        
        self.mean_probs = None
        self.uncertainty = None
        self.y_pred = None
        
    def _calculate_ece(self, y_true, y_prob, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        y_prob_max = np.max(y_prob, axis=1)
        y_pred = np.argmax(y_prob, axis=1)
        
        ece = 0.0
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
            in_bin = (y_prob_max > bin_lower) & (y_prob_max <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin] == y_pred[in_bin])
                avg_confidence_in_bin = np.mean(y_prob_max[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return float(ece)

    def execute_mc_dropout(self, num_passes=50):
        print(f"[{self.experiment_name}] Executing {num_passes} MC-Dropout forward passes...")
        predictions = []
        for _ in range(num_passes):
            predictions.append(self.model(self.X_test, training=True).numpy())
            
        predictions = np.array(predictions)
        self.mean_probs = np.mean(predictions, axis=0)
        
        variance_probs = np.var(predictions, axis=0)
        self.uncertainty = np.mean(variance_probs, axis=1) # Mean variance across classes
        self.y_pred = np.argmax(self.mean_probs, axis=1)

    def generate_report(self, save_dir="docs/reports"):
        if self.mean_probs is None:
            self.execute_mc_dropout()
            
        print(f"\n==================================================")
        print(f"RESEARCH EVALUATION PROTOCOL: {self.experiment_name}")
        print(f"==================================================")
        
        report = {
            "experiment_name": self.experiment_name,
            "data_integrity": {
                "test_cohort_size": int(len(self.y_test)),
                "class_distribution": {
                    "noise": int(np.sum(self.y_test == 0)),
                    "planet": int(np.sum(self.y_test == 1)),
                    "eb": int(np.sum(self.y_test == 2))
                }
            },
            "prediction": {},
            "calibration": {},
            "uncertainty": {}
        }
        
        # 1. Prediction (Discrimination)
        y_test_onehot = tf.keras.utils.to_categorical(self.y_test, num_classes=self.n_classes)
        acc = float(np.mean(self.y_test == self.y_pred))
        roc_auc = float(roc_auc_score(y_test_onehot, self.mean_probs, multi_class='ovr', average='weighted'))
        
        clf_report = classification_report(self.y_test, self.y_pred, target_names=self.class_names, output_dict=True)
        conf_matrix = confusion_matrix(self.y_test, self.y_pred).tolist()
        
        report["prediction"] = {
            "accuracy": acc,
            "roc_auc_weighted_ovr": roc_auc,
            "classification_report": clf_report,
            "confusion_matrix": conf_matrix
        }
        
        # 2. Calibration
        brier = float(np.mean(np.sum((self.mean_probs - y_test_onehot)**2, axis=1)))
        ece = self._calculate_ece(self.y_test, self.mean_probs)
        
        report["calibration"] = {
            "multiclass_brier_score": brier,
            "expected_calibration_error": ece
        }
        
        # 3. Uncertainty
        correct_mask = (self.y_test == self.y_pred)
        incorrect_mask = ~correct_mask
        
        unc_correct = float(np.mean(self.uncertainty[correct_mask])) if np.sum(correct_mask) > 0 else 0.0
        unc_incorrect = float(np.mean(self.uncertainty[incorrect_mask])) if np.sum(incorrect_mask) > 0 else 0.0
        
        report["uncertainty"] = {
            "mean_mc_variance_global": float(np.mean(self.uncertainty)),
            "mean_mc_variance_correct": unc_correct,
            "mean_mc_variance_incorrect": unc_incorrect
        }
        
        # Output to console
        print("\n--- 1. DATA INTEGRITY ---")
        print(f"Test Cohort Size: {report['data_integrity']['test_cohort_size']}")
        print(f"Class Distribution: Noise={report['data_integrity']['class_distribution']['noise']}, "
              f"Planet={report['data_integrity']['class_distribution']['planet']}, "
              f"EB={report['data_integrity']['class_distribution']['eb']}")
        
        print("\n--- 2. PREDICTION ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC (OVR): {roc_auc:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))
        print("Confusion Matrix:")
        print(np.array(conf_matrix))
        
        print("\n--- 3. CALIBRATION ---")
        print(f"Multiclass Brier Score: {brier:.4f}")
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        
        print("\n--- 4. UNCERTAINTY ---")
        print(f"Global MC Variance: {report['uncertainty']['mean_mc_variance_global']:.5f}")
        print(f"Correct Classification Variance: {unc_correct:.5f}")
        print(f"Incorrect Classification Variance: {unc_incorrect:.5f}")
        
        # Save JSON
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.experiment_name}_evaluation.json")
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\n[Saved] Full evaluation report written to {save_path}")
        
        return report

# Example Usage:
# evaluator = ResearchEvaluator("EXP6_NATIVE_MAD", "data/models/exp6_native_mad_model.keras", X_test_mad, y_test)
# evaluator.generate_report()
