#include "lstm_inference.h"
#include "temp_predictor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// LSTM state variables
static LSTMState lstm1_state = {0};
static LSTMState lstm2_state = {0};

void lstm_init(void) {
    // Initialize LSTM states
    lstm1_state.hidden_state = (float*)calloc(LSTM1_UNITS, sizeof(float));
    lstm1_state.cell_state = (float*)calloc(LSTM1_UNITS, sizeof(float));
    
    lstm2_state.hidden_state = (float*)calloc(LSTM2_UNITS, sizeof(float));
    lstm2_state.cell_state = (float*)calloc(LSTM2_UNITS, sizeof(float));
}

static float tanh_activation(float x) {
    return tanhf(x);
}

static float sigmoid_activation(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void lstm_layer(const float* input, const float* kernel, 
                      const float* recurrent_kernel, const float* bias,
                      float* hidden_state, float* cell_state,
                      int input_dim, int units, int return_sequences) {
    
    // Temporary arrays for gates
    float gates[4 * units];
    
    // Reset gates to bias values
    for (int i = 0; i < 4 * units; i++) {
        gates[i] = bias[i];
    }
    
    // Input * kernel
    for (int i = 0; i < 4 * units; i++) {
        for (int j = 0; j < input_dim; j++) {
            gates[i] += input[j] * kernel[j * (4 * units) + i];
        }
    }
    
    // Hidden state * recurrent_kernel
    for (int i = 0; i < 4 * units; i++) {
        for (int j = 0; j < units; j++) {
            gates[i] += hidden_state[j] * recurrent_kernel[j * (4 * units) + i];
        }
    }
    
    // Apply gates and update states
    for (int i = 0; i < units; i++) {
        float input_gate = sigmoid_activation(gates[i]);
        float forget_gate = sigmoid_activation(gates[units + i]);
        float cell_gate = tanh_activation(gates[2 * units + i]);
        float output_gate = sigmoid_activation(gates[3 * units + i]);
        
        // Update cell state
        cell_state[i] = forget_gate * cell_state[i] + input_gate * cell_gate;
        
        // Update hidden state
        hidden_state[i] = output_gate * tanh_activation(cell_state[i]);
    }
}

float predict_temperature(const float* input_sequence) {
    float current_input[INPUT_DIM];
    float lstm1_output[LSTM1_UNITS];
    
    // Reset LSTM states for new sequence
    memset(lstm1_state.hidden_state, 0, LSTM1_UNITS * sizeof(float));
    memset(lstm1_state.cell_state, 0, LSTM1_UNITS * sizeof(float));
    memset(lstm2_state.hidden_state, 0, LSTM2_UNITS * sizeof(float));
    memset(lstm2_state.cell_state, 0, LSTM2_UNITS * sizeof(float));
    
    // Process sequence through first LSTM layer (with return_sequences=True)
    for (int t = 0; t < SEQ_LEN; t++) {
        current_input[0] = input_sequence[t];
        
        // First LSTM layer (64 units)
        lstm_layer(current_input, weight_0, weight_1, weight_2,
                  lstm1_state.hidden_state, lstm1_state.cell_state,
                  INPUT_DIM, LSTM1_UNITS, 1);
        
        // Store output for the last time step only
        if (t == SEQ_LEN - 1) {
            for (int i = 0; i < LSTM1_UNITS; i++) {
                lstm1_output[i] = lstm1_state.hidden_state[i];
            }
        }
    }
    
    // Second LSTM layer (32 units, return_sequences=False)
    lstm_layer(lstm1_output, weight_3, weight_4, weight_5,
              lstm2_state.hidden_state, lstm2_state.cell_state,
              LSTM1_UNITS, LSTM2_UNITS, 0);
    
    // First Dense layer (16 units)
    float dense1_output[16] = {0};
    for (int i = 0; i < 16; i++) {
        dense1_output[i] = 0;
        for (int j = 0; j < LSTM2_UNITS; j++) {
            dense1_output[i] += lstm2_state.hidden_state[j] * weight_6[j * 16 + i];
        }
        dense1_output[i] += weight_7[i]; // bias
        // ReLU activation
        if (dense1_output[i] < 0) dense1_output[i] = 0;
    }
    
    // Second Dense layer (8 units)
    float dense2_output[8] = {0};
    for (int i = 0; i < 8; i++) {
        dense2_output[i] = 0;
        for (int j = 0; j < 16; j++) {
            dense2_output[i] += dense1_output[j] * weight_8[j * 8 + i];
        }
        dense2_output[i] += weight_9[i]; // bias
        // ReLU activation
        if (dense2_output[i] < 0) dense2_output[i] = 0;
    }
    
    // Output layer (1 unit)
    float output = 0;
    for (int j = 0; j < 8; j++) {
        output += dense2_output[j] * weight_10[j];
    }
    output += weight_11[0]; // bias
    
    return output;
}

void lstm_cleanup(void) {
    free(lstm1_state.hidden_state);
    free(lstm1_state.cell_state);
    free(lstm2_state.hidden_state);
    free(lstm2_state.cell_state);
}