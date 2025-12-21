#ifndef LSTM_INFERENCE_H
#define LSTM_INFERENCE_H

#include <stddef.h>

typedef struct {
    float* hidden_state;
    float* cell_state;
} LSTMState;

void lstm_init(void);

float predict_temperature(const float* input_sequence);

void lstm_cleanup(void);


#endif
