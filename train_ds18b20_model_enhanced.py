#!/usr/bin/env python3
"""
train_ds18b20_model_with_plots_enhanced.py

Enhanced LSTM training for DS18B20 temperature data with better validation,
evaluation, and error handling.

Usage:
    pip install tensorflow scikit-learn joblib pandas numpy matplotlib
    python train_ds18b20_model_with_plots_enhanced.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------- CONFIG ----------
DATA_FILE = 'ds18b20_data.csv'
SEQ_LEN = 60
EPOCHS = 200
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
MODEL_FILE = 'ds18b20_model.keras'
SCALER_FILE = 'scaler.save'
TFLITE_FILE_F16 = 'ds18b20_model_float16.tflite'
TFLITE_FILE_DEFAULT = 'ds18b20_model_default.tflite'
TRAIN_PLOT = 'training_curves.png'
PREDICTIONS_PLOT = 'predictions_plot.png'
RESULTS_FILE = 'training_results.txt'
# ----------------------------

def find_temp_column(df):
    """Find temperature column in dataframe"""
    for col in df.columns:
        if 'temp' in col.lower():
            return col
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numcols) >= 2:
        return numcols[1]
    return numcols[0] if numcols else None

def enhanced_data_validation(df, temp_col):
    """Check data quality and remove extreme outliers"""
    print(f"Original data range: {df[temp_col].min():.2f} to {df[temp_col].max():.2f}")
    print(f"Data mean: {df[temp_col].mean():.2f}")
    print(f"Standard deviation: {df[temp_col].std():.2f}")
    print(f"Missing values: {df[temp_col].isna().sum()}")
    print(f"Total samples: {len(df)}")
    
    # Remove extreme outliers (beyond 4 standard deviations for more tolerance)
    mean = df[temp_col].mean()
    std = df[temp_col].std()
    df_clean = df[(df[temp_col] >= mean - 4*std) & (df[temp_col] <= mean + 4*std)].copy()
    
    if len(df) != len(df_clean):
        print(f"Removed {len(df) - len(df_clean)} outliers")
        print(f"Cleaned data range: {df_clean[temp_col].min():.2f} to {df_clean[temp_col].max():.2f}")
    
    return df_clean

def load_and_prepare():
    """Load and validate data"""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found in current folder.")
    
    df = pd.read_csv(DATA_FILE)
    temp_col = find_temp_column(df)
    
    if temp_col is None:
        raise ValueError("No suitable temperature column found in data")
    
    print(f"Using temperature column: {temp_col}")
    df = df[[temp_col]].dropna().reset_index(drop=True)
    
    # Enhanced data validation
    df_clean = enhanced_data_validation(df, temp_col)
    
    return df_clean, temp_col

def create_sequences(scaled, seq_len):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

def create_sequences_with_split(scaled, seq_len, train_ratio=0.8):
    """Create sequences and split properly to avoid data leakage"""
    X, y = create_sequences(scaled, seq_len)
    
    # Split maintaining temporal order
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training sequences: {X_train.shape}, Validation: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def build_tflite_compatible_model(seq_len):
    """Build LSTM model that's compatible with TFLite conversion"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        
        LSTM(32, return_sequences=False),  # Set return_sequences=False for last LSTM
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse', 
        metrics=['mae']
    )
    return model

def enhanced_plot_history(history, out_file):
    """More comprehensive training visualization"""
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    ax1.plot(epochs, hist['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in hist:
        ax1.plot(epochs, hist['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.plot(epochs, hist['mae'], label='Train MAE', linewidth=2)
    if 'val_mae' in hist:
        ax2.plot(epochs, hist['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in hist:
        ax3.semilogy(epochs, hist['lr'], linewidth=2, color='green')
        ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'Learning Rate data not available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    # Training summary - FIXED F-STRING SYNTAX
    final_train_loss = hist['loss'][-1]
    final_val_loss = hist.get('val_loss', ['N/A'])[-1] if 'val_loss' in hist else 'N/A'
    final_train_mae = hist['mae'][-1] if 'mae' in hist else 'N/A'
    final_val_mae = hist.get('val_mae', ['N/A'])[-1] if 'val_mae' in hist else 'N/A'
    
    # Format values properly
    final_train_loss_str = f"{final_train_loss:.6f}" if isinstance(final_train_loss, float) else str(final_train_loss)
    final_val_loss_str = f"{final_val_loss:.6f}" if isinstance(final_val_loss, float) else str(final_val_loss)
    final_train_mae_str = f"{final_train_mae:.6f}" if isinstance(final_train_mae, float) else str(final_train_mae)
    final_val_mae_str = f"{final_val_mae:.6f}" if isinstance(final_val_mae, float) else str(final_val_mae)
    
    summary_text = f"""Training Summary:
Epochs: {len(hist['loss'])}
Final Train Loss: {final_train_loss_str}
Final Val Loss: {final_val_loss_str}
Final Train MAE: {final_train_mae_str}
Final Val MAE: {final_val_mae_str}"""
    
    ax4.axis('off')
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', 
             transform=ax4.transAxes, fontsize=11, family='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved enhanced training curves to {out_file}")

def evaluate_model(model, X_val, y_val, scaler):
    """Comprehensive model evaluation"""
    print("Evaluating model on validation data...")
    predictions = model.predict(X_val, verbose=0)
    
    # Inverse transform to original scale
    y_true_orig = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(predictions).flatten()
    
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    
    # Calculate mean absolute percentage error (avoid division by zero)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(np.abs(y_true_orig), 1e-8))) * 100
    
    print(f"Validation MAE: {mae:.4f}°C")
    print(f"Validation RMSE: {rmse:.4f}°C")
    print(f"Validation MAPE: {mape:.2f}%")
    print(f"Average Temperature: {y_true_orig.mean():.2f}°C")
    
    return mae, rmse, mape, y_true_orig, y_pred_orig

def plot_predictions(y_true, y_pred, out_file='predictions_plot.png'):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    
    # Plot first 200 points for clarity
    plot_points = min(200, len(y_true))
    indices = range(plot_points)
    
    plt.plot(indices, y_true[:plot_points], label='Actual', linewidth=2, alpha=0.8, color='blue')
    plt.plot(indices, y_pred[:plot_points], label='Predicted', linewidth=2, alpha=0.8, linestyle='--', color='red')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.title('Actual vs Predicted Temperatures (First 200 Points)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions plot to {out_file}")

def safe_tflite_conversion(keras_model_path, out_path_f16, out_path_default):
    """More robust TFLite conversion with fallbacks"""
    try:
        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(f"Model file {keras_model_path} not found")
        
        print("Loading model for TFLite conversion...")
        model = tf.keras.models.load_model(keras_model_path)
        
        # Attempt 1: Float16 quantization with SELECT_TF_OPS
        print("Attempt 1: Float16 quantization with SELECT_TF_OPS...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
            
            tflite_model = converter.convert()
            
            with open(out_path_f16, 'wb') as f:
                f.write(tflite_model)
            print(f"✓ Successfully converted to FLOAT16 with SELECT_TF_OPS: {out_path_f16}")
            return True
            
        except Exception as e1:
            print(f"Float16 conversion failed: {e1}")
            
            # Attempt 2: Default quantization with SELECT_TF_OPS
            print("Attempt 2: Default quantization with SELECT_TF_OPS...")
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                converter._experimental_lower_tensor_list_ops = False
                
                tflite_model = converter.convert()
                
                with open(out_path_default, 'wb') as f:
                    f.write(tflite_model)
                print(f"✓ Successfully converted to DEFAULT with SELECT_TF_OPS: {out_path_default}")
                return True
                
            except Exception as e2:
                print(f"Default conversion with SELECT_TF_OPS failed: {e2}")
                
                # Attempt 3: Simple conversion without optimizations
                print("Attempt 3: Simple conversion without optimizations...")
                try:
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    tflite_model = converter.convert()
                    
                    simple_path = out_path_default.replace('.tflite', '_simple.tflite')
                    with open(simple_path, 'wb') as f:
                        f.write(tflite_model)
                    print(f"✓ Successfully converted to SIMPLE: {simple_path}")
                    return True
                    
                except Exception as e3:
                    print(f"All conversion attempts failed: {e3}")
                    return False
                    
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        return False

def save_training_results(history, metrics, file_path):
    """Save training results to text file"""
    hist = history.history
    final_epoch = len(hist['loss'])
    
    with open(file_path, 'w') as f:
        f.write("DS18B20 LSTM Training Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Training completed at epoch: {final_epoch}\n")
        f.write(f"Final training loss: {hist['loss'][-1]:.6f}\n")
        
        if 'val_loss' in hist:
            f.write(f"Final validation loss: {hist['val_loss'][-1]:.6f}\n")
        
        f.write(f"Final training MAE: {hist['mae'][-1]:.6f}\n")
        
        if 'val_mae' in hist:
            f.write(f"Final validation MAE: {hist['val_mae'][-1]:.6f}\n")
        
        f.write("\nValidation Metrics:\n")
        f.write(f"MAE: {metrics[0]:.4f}°C\n")
        f.write(f"RMSE: {metrics[1]:.4f}°C\n")
        f.write(f"MAPE: {metrics[2]:.2f}%\n")
        
        f.write(f"\nModel Configuration:\n")
        f.write(f"Sequence length: {SEQ_LEN}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Training ratio: {TRAIN_RATIO}\n")
        
        f.write(f"\nDataset Info:\n")
        f.write(f"Total samples: {metrics[3] if len(metrics) > 3 else 'N/A'}\n")

def main():
    print("Starting enhanced DS18B20 LSTM training...")
    
    try:
        # Load and prepare data
        df, temp_col = load_and_prepare()
        print(f"Loaded samples after cleaning: {len(df)}")

        # Scale data
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[[temp_col]].values.astype(float))
        joblib.dump(scaler, SCALER_FILE)
        print(f"Saved scaler to {SCALER_FILE}")

        # Create sequences with proper split
        X_train, X_val, y_train, y_val = create_sequences_with_split(
            scaled, SEQ_LEN, TRAIN_RATIO
        )

        # Build TFLite compatible model
        model = build_tflite_compatible_model(SEQ_LEN)
        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
        ]

        # Train model
        print("Starting training...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series data
        )

        # Save final model using new Keras format
        model.save(MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")

        # Enhanced plots
        enhanced_plot_history(history, TRAIN_PLOT)

        # Evaluate model
        mae, rmse, mape, y_true_orig, y_pred_orig = evaluate_model(
            model, X_val, y_val, scaler
        )
        
        # Plot predictions
        plot_predictions(y_true_orig, y_pred_orig, PREDICTIONS_PLOT)

        # Save training results
        save_training_results(history, (mae, rmse, mape, len(df)), RESULTS_FILE)
        print(f"Training results saved to {RESULTS_FILE}")

        # TFLite conversion
        print("\nStarting TFLite conversion...")
        conversion_success = safe_tflite_conversion(
            MODEL_FILE, 
            TFLITE_FILE_F16, 
            TFLITE_FILE_DEFAULT
        )
        
        if conversion_success:
            print("✓ TFLite conversion completed successfully")
        else:
            print("✗ TFLite conversion failed")

        print("\n" + "="*50)
        print("Training completed successfully!")
        print("Generated files:")
        print(f"  - Model: {MODEL_FILE}")
        print(f"  - Scaler: {SCALER_FILE}")
        if os.path.exists(TFLITE_FILE_F16):
            print(f"  - TFLite (Float16): {TFLITE_FILE_F16}")
        if os.path.exists(TFLITE_FILE_DEFAULT):
            print(f"  - TFLite (Default): {TFLITE_FILE_DEFAULT}")
        print(f"  - Training curves: {TRAIN_PLOT}")
        print(f"  - Predictions plot: {PREDICTIONS_PLOT}")
        print(f"  - Results: {RESULTS_FILE}")
        print("="*50)

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()