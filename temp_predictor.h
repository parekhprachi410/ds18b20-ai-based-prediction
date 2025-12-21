#ifndef TEMP_PREDICTOR_H
#define TEMP_PREDICTOR_H

#define SEQ_LEN 60
#define INPUT_DIM 1
#define LSTM1_UNITS 64
#define LSTM2_UNITS 32
#define OUTPUT_DIM 1

// Model weights and biases
extern const float weight_0[256];
extern const float weight_1[16384];
extern const float weight_2[256];
extern const float weight_3[8192];
extern const float weight_4[4096];
extern const float weight_5[128];
extern const float weight_6[512];
extern const float weight_7[16];
extern const float weight_8[128];
extern const float weight_9[8];
extern const float weight_10[8];
extern const float weight_11[1];

// Weight array pointers
extern const float* model_weights[];
extern const int num_weight_arrays;
extern const int weight_sizes[];

#endif 

