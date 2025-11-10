#ifndef SCALER_H
#define SCALER_H

// Scaler parameters from MinMaxScaler training
#define SCALER_DATA_MIN 7.312f
#define SCALER_DATA_MAX 38.938f  
#define SCALER_SCALE 0.03161955f
#define SCALER_MIN -0.23120218f

float scale_input(float temperature);
float scale_output(float scaled_output);

#endif