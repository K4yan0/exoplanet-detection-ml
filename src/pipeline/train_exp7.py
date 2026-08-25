import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def build_model(input_shape=(2000, 1)):
    # V1 Reference Architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save(dataset_path, model_name):
    print(f"\n==================================================")
    print(f"TRAINING EXP 7: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"==================================================")
    
    data = np.load(dataset_path)
    X = data['X']
    Y = data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train)
    
    # Class weights for imbalance
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    model = build_model()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )
    
    os.makedirs('data/models', exist_ok=True)
    save_path = f"data/models/{model_name}.keras"
    model.save(save_path)
    print(f"Model saved to {save_path}")

def main():
    path_1sec = 'data/tess_ml_arrays/tess_dataset_exp7_1sec.npz'
    path_5sec = 'data/tess_ml_arrays/tess_dataset_exp7_5sec.npz'
    
    train_and_save(path_1sec, "exp7_1sec_model")
    train_and_save(path_5sec, "exp7_5sec_model")

if __name__ == '__main__':
    main()
