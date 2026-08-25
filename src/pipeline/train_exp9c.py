import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

def build_exp9c_model():
    """
    Exp 9C Architecture (Position-Preserving Low-Dimensional Fusion)
    
    Local Branch:
    - Input: (5, 2000, 1) independent phase-folded sectors.
    - Encoder: Shared CNN with heavy strided pooling to reduce to 20 spatial bins.
    - Compression: Flatten (to preserve coarse position) instead of GlobalAveragePooling1D.
    - Aggregation: LSTM to learn cross-sector phase drifts.
    - Classifier: Dense head.
    """
    
    local_input = tf.keras.Input(shape=(5, 2000, 1), name='local_morphology_input')
    
    shared_cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2000, 1)),
        tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 1000
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2), # 500
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=5), # 100
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=5), # 20
        
        # Flatten preserves the 20 coarse positional bins!
        # (20 bins * 64 channels = 1280 features)
        tf.keras.layers.Flatten(), 
        
        # Low-dimensional bottleneck before LSTM
        tf.keras.layers.Dense(128, activation='relu')
    ], name='shared_sector_cnn_coarse')
    
    encoded_sectors = tf.keras.layers.TimeDistributed(shared_cnn)(local_input) # (batch, 5, 128)
    
    # RECURRENT SEQUENCE MODELING
    sequence_embedding = tf.keras.layers.LSTM(64)(encoded_sectors)  # (batch, 64)
    
    # --- Classifier ---
    x = tf.keras.layers.Dropout(0.3)(sequence_embedding)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=local_input, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print(f"\n==================================================")
    print(f"TRAINING EXP 9C: Position-Preserving LSTM Fusion")
    print(f"==================================================")
    
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
        
    data = np.load(dataset_path)
    X = data['X']
    Y = data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    model = build_exp9c_model()
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f}")
    
    os.makedirs('data/models', exist_ok=True)
    save_path = 'data/models/exp9c_model.keras'
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
