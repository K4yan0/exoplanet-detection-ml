import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def build_exp8_model(input_shape=(5, 2000, 1)):
    """
    Exp 8 Architecture:
    - Input: 5 independent phase-folded sectors.
    - Encoder: Shared CNN applied to each sector via TimeDistributed.
    - Aggregation: Mean pooling of the 5 sector embeddings.
    - Classifier: Dense head.
    """
    # 1. The Shared CNN Encoder
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2000, 1)),
        tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ], name='shared_sector_encoder')
    
    # 2. The Global Model
    inputs = tf.keras.Input(shape=input_shape)
    
    # Extract independent embeddings for each of the 5 sectors
    encoded_sectors = tf.keras.layers.TimeDistributed(encoder)(inputs) # Shape: (batch, 5, 128)
    
    # Aggregate embeddings (mean pooling)
    fused_representation = tf.keras.layers.GlobalAveragePooling1D()(encoded_sectors) # Shape: (batch, 128)
    
    # 3. The Classifier Head
    x = tf.keras.layers.Dropout(0.3)(fused_representation)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print(f"\n==================================================")
    print(f"TRAINING EXP 8: Independent Sector Aggregation")
    print(f"==================================================")
    
    dataset_path = 'data/tess_ml_arrays/tess_dataset_exp8.npz'
    if not os.path.exists(dataset_path):
        print("Dataset not found. Run build_exp8_dataset.py first.")
        return
        
    data = np.load(dataset_path)
    X = data['X']
    Y = data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train)
    
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    model = build_exp8_model()
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
    save_path = 'data/models/exp8_model.keras'
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
