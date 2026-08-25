import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# We need to provide the custom layers to load the model
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

class AttentionPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(1, use_bias=False, name='attention_score')
        super(AttentionPooling1D, self).build(input_shape)

    def call(self, inputs):
        scores = self.attention_dense(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        weighted_inputs = inputs * weights
        pooled = tf.reduce_sum(weighted_inputs, axis=1)
        return pooled, weights

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == 1)
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def mc_dropout_predict(model, X, num_samples=50):
    # Enable dropout during inference by setting training=True
    predictions = []
    for _ in range(num_samples):
        predictions.append(model(X, training=True))
    
    predictions = np.array(predictions)
    mean_preds = np.mean(predictions, axis=0)
    std_preds = np.std(predictions, axis=0)
    return mean_preds, std_preds

def evaluate_model():
    model_path = 'data/models/exp10b_model.keras'
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    
    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        print("Model or dataset missing.")
        return

    # Load custom objects
    custom_objects = {
        'TransformerBlock': TransformerBlock,
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
        'AttentionPooling1D': AttentionPooling1D
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    data = np.load(dataset_path)
    X = data['X']
    Y = data['y']
    tics = data['tics']
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    print("--- Exp 10B (Self-Attention + Attention Pooling) Full Evaluation ---\n")
    
    # 1. Standard Metrics
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print("1. Overall Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Noise', 'Planet', 'EB']))
    
    print("2. Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred))
    
    roc_auc = roc_auc_score(tf.keras.utils.to_categorical(y_test), y_pred_prob, multi_class='ovr')
    print(f"\nMacro ROC-AUC: {roc_auc:.4f}")
    
    y_test_planet = (y_test == 1).astype(int)
    y_prob_planet = y_pred_prob[:, 1]
    brier = brier_score_loss(y_test_planet, y_prob_planet)
    ece = expected_calibration_error(y_test_planet, y_prob_planet)
    
    print(f"Brier Score (Planet): {brier:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # 2. Targeted Recovery of 10 Planet Diagnoses
    diagnostics_list = [
        "TIC TIC 259377017_Positive",
        "TIC TIC 36724087_Positive",
        "TIC TIC 287328202_Positive",
        "TIC TIC 345143460_Positive",
        "TIC TIC 234994474_Positive",
        "TIC TIC 150030205_Positive",
        "TIC TIC 262530407_Positive",
        "TIC TIC 181804752_Positive",
        "TIC TIC 307809773_Positive",
        "TIC TIC 254113311_Positive"
    ]
    
    print("\n--- Diagnostic: Recovery of the 10 Targeted Planets ---")
    
    # Generate mapping from TIC to index
    tic_to_idx = {tic_str: i for i, tic_str in enumerate(tics)}
    
    recovered = 0
    for diag_tic in diagnostics_list:
        if diag_tic in tic_to_idx:
            idx = tic_to_idx[diag_tic]
            x_diag = X[idx:idx+1] # Keep batch dimension
            
            # Use MC Dropout for confidence
            mean_probs, std_probs = mc_dropout_predict(model, x_diag, num_samples=100)
            
            p_planet_mean = mean_probs[0, 1]
            p_planet_std = std_probs[0, 1]
            pred_class = np.argmax(mean_probs[0])
            
            if pred_class == 1:
                status = "RECOVERED"
                recovered += 1
                print(f"{diag_tic}: {status} | P(Planet) = {p_planet_mean:.3f}  {p_planet_std:.3f}")
            else:
                status = "MISSED   "
                print(f"{diag_tic}: {status} | Predicted: {pred_class} | P(Planet) = {p_planet_mean:.3f}  {p_planet_std:.3f}")
        else:
            print(f"{diag_tic}: NOT FOUND IN DATASET")
            
    print(f"\nTotal Recovered: {recovered}/10")
    
if __name__ == "__main__":
    evaluate_model()
