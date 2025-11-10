#include "scaler.h"

float scale_input(float temperature) {
    return (temperature - SCALER_DATA_MIN) * SCALER_SCALE + SCALER_MIN;
}

float scale_output(float scaled_output) {
    return (scaled_output - SCALER_MIN) / SCALER_SCALE + SCALER_DATA_MIN;
}