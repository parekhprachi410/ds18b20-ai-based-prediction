#ifndef LSTM_INFERENCE_H
#define LSTM_INFERENCE_H

#include <stddef.h>

// LSTM inference functions for ESP32
typedef struct {
    float* hidden_state;
    float* cell_state;
} LSTMState;

// Initialize LSTM inference
void lstm_init(void);

// Single prediction function
float predict_temperature(const float* input_sequence);

// Cleanup function
void lstm_cleanup(void);

#endif