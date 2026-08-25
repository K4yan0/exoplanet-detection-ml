import numpy as np
from sklearn.model_selection import train_test_split
from research_evaluator import ResearchEvaluator

RANDOM_SEED = 42

def main():
    path_1sec = 'data/tess_ml_arrays/tess_dataset_exp7_1sec.npz'
    path_5sec = 'data/tess_ml_arrays/tess_dataset_exp7_5sec.npz'
    
    # Load test sets strictly sequestered exactly as during training
    d1 = np.load(path_1sec)
    X_1sec, y_1 = d1['X'], d1['y']
    _, X_test_1sec, _, y_test = train_test_split(X_1sec, y_1, test_size=0.2, random_state=RANDOM_SEED, stratify=y_1)
    
    d5 = np.load(path_5sec)
    X_5sec, y_5 = d5['X'], d5['y']
    _, X_test_5sec, _, _ = train_test_split(X_5sec, y_5, test_size=0.2, random_state=RANDOM_SEED, stratify=y_5)
    
    model_1_path = 'data/models/exp7_1sec_model.keras'
    model_5_path = 'data/models/exp7_5sec_model.keras'
    
    # 2x2 Matrix Evaluation
    
    print("\n\n" + "="*80)
    print("CELL 1: 1-SECTOR CNN on 1-SECTOR INPUT (Baseline)")
    print("="*80)
    eval_11 = ResearchEvaluator("EXP7_1CNN_1IN", model_1_path, X_test_1sec, y_test)
    rep_11 = eval_11.generate_report()
    
    print("\n\n" + "="*80)
    print("CELL 2: 1-SECTOR CNN on 5-SECTOR INPUT (Inference Shift)")
    print("="*80)
    eval_15 = ResearchEvaluator("EXP7_1CNN_5IN", model_1_path, X_test_5sec, y_test)
    rep_15 = eval_15.generate_report()
    
    print("\n\n" + "="*80)
    print("CELL 3: 5-SECTOR CNN on 1-SECTOR INPUT (Inference Shift)")
    print("="*80)
    eval_51 = ResearchEvaluator("EXP7_5CNN_1IN", model_5_path, X_test_1sec, y_test)
    rep_51 = eval_51.generate_report()
    
    print("\n\n" + "="*80)
    print("CELL 4: 5-SECTOR CNN on 5-SECTOR INPUT (Native)")
    print("="*80)
    eval_55 = ResearchEvaluator("EXP7_5CNN_5IN", model_5_path, X_test_5sec, y_test)
    rep_55 = eval_55.generate_report()
    
    # Matrix Summary
    print("\n\n==================================================")
    print("EXP 7 SUMMARY MATRIX (Accuracy / ECE / MC-Var)")
    print("==================================================")
    print(f"               | 1-Sector Input                 | 5-Sector Input                 |")
    print(f"---------------|--------------------------------|--------------------------------|")
    print(f"1-Sector CNN   | {rep_11['prediction']['accuracy']:.4f} / {rep_11['calibration']['expected_calibration_error']:.4f} / {rep_11['uncertainty']['mean_mc_variance_global']:.4f} | {rep_15['prediction']['accuracy']:.4f} / {rep_15['calibration']['expected_calibration_error']:.4f} / {rep_15['uncertainty']['mean_mc_variance_global']:.4f} |")
    print(f"5-Sector CNN   | {rep_51['prediction']['accuracy']:.4f} / {rep_51['calibration']['expected_calibration_error']:.4f} / {rep_51['uncertainty']['mean_mc_variance_global']:.4f} | {rep_55['prediction']['accuracy']:.4f} / {rep_55['calibration']['expected_calibration_error']:.4f} / {rep_55['uncertainty']['mean_mc_variance_global']:.4f} |")

if __name__ == '__main__':
    main()
