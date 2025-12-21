import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_FILE = 'ds18b20_data.csv'
SEQ_LEN = 60
EPOCHS = 200
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
MODEL_FILE = 'ds18b20_model.keras'
SCALER_FILE = 'scaler.save'
TFLITE_F16 = 'ds18b20_model_float16.tflite'
TFLITE_DEFAULT = 'ds18b20_model_default.tflite'
TRAIN_PLOT = 'training_curves.png'
PREDICT_PLOT = 'predictions_plot.png'
RESULTS_FILE = 'training_results.txt'

def get_temp_col(df):
    for col in df.columns:
        if 'temp' in col.lower():
            return col
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[1] if len(nums) >= 2 else (nums[0] if nums else None)

def clean_data(df, col):
    print(f"Data stats -> min: {df[col].min():.2f}, max: {df[col].max():.2f}, mean: {df[col].mean():.2f}, std: {df[col].std():.2f}")
    
    mean, std = df[col].mean(), df[col].std()
    cleaned = df[(df[col] >= mean - 4*std) & (df[col] <= mean + 4*std)].copy()
    
    removed = len(df) - len(cleaned)
    if removed > 0:
        print(f"Removed {removed} extreme outliers. Clean range: {cleaned[col].min():.2f}-{cleaned[col].max():.2f}")
    return cleaned

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found!")
    df = pd.read_csv(DATA_FILE)
    col = get_temp_col(df)
    if col is None:
        raise ValueError("No numeric/temperature column found!")
    print(f"Using '{col}' as temperature column")
    df = df[[col]].dropna().reset_index(drop=True)
    return clean_data(df, col), col

def make_seq(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def split_seq(data, seq_len, train_ratio=0.8):
    X, y = make_seq(data, seq_len)
    split = int(len(X) * train_ratio)
    print(f"Sequences -> train: {X[:split].shape}, val: {X[split:].shape}")
    return X[:split], X[split:], y[:split], y[split:]

def build_model(seq_len):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len,1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def plot_hist(history, file):
    hist = history.history
    epochs = range(1, len(hist['loss'])+1)
    plt.figure(figsize=(12,5))
    plt.plot(epochs, hist['loss'], label='train loss')
    plt.plot(epochs, hist.get('val_loss', []), label='val loss')
    plt.plot(epochs, hist['mae'], label='train mae')
    plt.plot(epochs, hist.get('val_mae', []), label='val mae')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(file, dpi=150)
    plt.close()
    print(f"Saved training curves -> {file}")

def eval_model(model, X_val, y_val, scaler):
    pred = model.predict(X_val, verbose=0)
    y_true = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
    y_pred = scaler.inverse_transform(pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true),1e-8))) * 100
    print(f"Validation -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")
    return mae, rmse, mape, y_true, y_pred

def plot_pred(y_true, y_pred, file):
    plt.figure(figsize=(12,5))
    points = min(200, len(y_true))
    plt.plot(range(points), y_true[:points], label='Actual', color='blue')
    plt.plot(range(points), y_pred[:points], label='Predicted', color='red', linestyle='--')
    plt.title('Actual vs Predicted Temps')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(file, dpi=150)
    plt.close()
    print(f"Saved prediction plot -> {file}")

def to_tflite(model_file, f16_file, default_file):
    if not os.path.exists(model_file):
        print(f"{model_file} not found, skipping TFLite conversion")
        return
    print("Converting to TFLite...")
    model = tf.keras.models.load_model(model_file)
    for file, dtype in [(f16_file, tf.float16), (default_file, None)]:
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if dtype: 
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [dtype]
            tflite_model = converter.convert()
            with open(file, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite saved -> {file}")
        except Exception as e:
            print(f"Conversion to {file} failed: {e}")

def train():
    print("=== DS18B20 LSTM Training ===")
    try:
        df, col = load_data()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values.astype(float))
        joblib.dump(scaler, SCALER_FILE)
        print(f"Scaler saved -> {SCALER_FILE}")
        
        X_train, X_val, y_train, y_val = split_seq(scaled, SEQ_LEN, TRAIN_RATIO)
        model = build_model(SEQ_LEN)
        model.summary()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=1,
                            shuffle=False)
        model.save(MODEL_FILE)
        print(f"Model saved -> {MODEL_FILE}")
        
        plot_hist(history, TRAIN_PLOT)
        mae, rmse, mape, y_true, y_pred = eval_model(model, X_val, y_val, scaler)
        plot_pred(y_true, y_pred, PREDICT_PLOT)
        to_tflite(MODEL_FILE, TFLITE_F16, TFLITE_DEFAULT)
        
    except Exception as e:
        print("Oops, something went wrong!")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    train()
