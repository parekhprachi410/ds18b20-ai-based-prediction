#include "temp_predictor.h"
#include "lstm_inference.h"
#include "scaler.h"
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Arduino.h>

#ifndef SEQ_LEN
#define SEQ_LEN 60
#endif

#define ONE_WIRE_BUS 4
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

const unsigned long PREDICTION_INTERVAL = 1000UL;
const float SPIKE_THRESHOLD = 0.6f;
const int FAST_ADOPT_STEPS = 5;
const float BLEND_BETA = 0.65f;
const bool ENABLE_LINEAR_CORRECTION = true;
const float BIAS_ALPHA = 0.02f;
const float PARAM_ALPHA = 0.3f;
const float STABLE_DELTA_THRESH = 0.15f;
const float SLOPE_MIN = 0.7f;
const float SLOPE_MAX = 1.3f;
const float EPS_NUM = 1e-6f;

float temperature_history[SEQ_LEN];
int history_index = 0;
int history_count = 0;

bool sensor_connected = true;
float last_temperature = 22.0f;
unsigned long last_prediction = 0;
int fast_adopt_counter = 0;

// Linear correction state
float ema_M = 0.0f;
float ema_S = 0.0f;
float ema_MS = 0.0f;
float ema_M2 = 0.0f;
bool linear_initialized = false;
float learned_slope = 1.0f;
float learned_intercept = 0.0f;

// Forward declarations
float make_prediction();
float read_temperature();
void simulate_temperature_variation();
void update_linear_correction(float model_pred, float measured, bool force_update=false);
float apply_linear_correction(float model_pred);
void reset_linear_params();

void setup() {
  Serial.begin(115200);
  delay(1000);
  sensors.begin();
  sensors.setResolution(12);

  int deviceCount = sensors.getDeviceCount();
  sensor_connected = (deviceCount > 0);

  Serial.print("real time: ");
  Serial.println(sensor_connected ? "connected" : "disconnected");

  lstm_init();

  float initial_temp = read_temperature();
  for (int i = 0; i < SEQ_LEN; i++) temperature_history[i] = initial_temp;
  history_index = 0; history_count = SEQ_LEN;
  last_temperature = initial_temp;

  reset_linear_params();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'R') {
      reset_linear_params();
      Serial.println("Linear params reset by user.");
    }
  }

  unsigned long now = millis();
  if (now - last_prediction < PREDICTION_INTERVAL) return;
  last_prediction = now;

  float current_temperature = read_temperature();
  float final_pred = current_temperature;

  if (sensor_connected) {
    // Spike handling
    float delta = current_temperature - last_temperature;
    if (fabs(delta) >= SPIKE_THRESHOLD) {
      for (int i = 0; i < SEQ_LEN; i++) temperature_history[i] = current_temperature;
      history_index = 0; history_count = SEQ_LEN; fast_adopt_counter = FAST_ADOPT_STEPS;
      final_pred = current_temperature;
    } else {
      // Normal update
      temperature_history[history_index] = current_temperature;
      history_index = (history_index + 1) % SEQ_LEN;
      if (history_count < SEQ_LEN) history_count++;

      float model_pred = make_prediction();
      if (ENABLE_LINEAR_CORRECTION && fabs(current_temperature - last_temperature) < STABLE_DELTA_THRESH && history_count >= SEQ_LEN) {
        update_linear_correction(model_pred, current_temperature);
      }
      float model_corr = apply_linear_correction(model_pred);
      final_pred = BLEND_BETA * model_corr + (1.0f - BLEND_BETA) * current_temperature;
    }
  } else {
    // Sensor disconnected
    simulate_temperature_variation();
    float model_pred = (history_count >= SEQ_LEN) ? make_prediction() : current_temperature;
    float model_corr = ENABLE_LINEAR_CORRECTION ? apply_linear_correction(model_pred) : model_pred;
    final_pred = BLEND_BETA * model_corr + (1.0f - BLEND_BETA) * last_temperature;
  }

  last_temperature = current_temperature;

  Serial.print("real time: ");
  Serial.println(sensor_connected ? "connected" : "disconnected");
  Serial.print("temperature= ");
  Serial.println(current_temperature, 2);
  Serial.print("prediction= ");
  Serial.println(final_pred, 2);
}

// Linear correction functions
void update_linear_correction(float model_pred, float measured, bool force_update) {
  if (!linear_initialized) {
    ema_M = model_pred; ema_S = measured; ema_MS = model_pred*measured; ema_M2 = model_pred*model_pred;
    learned_slope = 1.0f; learned_intercept = 0.0f; linear_initialized = true;
    return;
  }

  ema_M  = (1.0f - BIAS_ALPHA) * ema_M  + BIAS_ALPHA * model_pred;
  ema_S  = (1.0f - BIAS_ALPHA) * ema_S  + BIAS_ALPHA * measured;
  ema_MS = (1.0f - BIAS_ALPHA) * ema_MS + BIAS_ALPHA * (model_pred * measured);
  ema_M2 = (1.0f - BIAS_ALPHA) * ema_M2 + BIAS_ALPHA * (model_pred * model_pred);

  float denom = (ema_M2 - ema_M * ema_M);
  float slope = (fabs(denom) > EPS_NUM) ? (ema_MS - ema_M * ema_S) / denom : 1.0f;
  slope = constrain(slope, SLOPE_MIN, SLOPE_MAX);
  float intercept = ema_S - slope * ema_M;

  learned_slope = (1.0f - PARAM_ALPHA) * learned_slope + PARAM_ALPHA * slope;
  learned_intercept = (1.0f - PARAM_ALPHA) * learned_intercept + PARAM_ALPHA * intercept;
}

float apply_linear_correction(float model_pred) {
  return learned_slope * model_pred + learned_intercept;
}

void reset_linear_params() {
  linear_initialized = false;
  learned_slope = 1.0f;
  learned_intercept = 0.0f;
  ema_M = last_temperature;
  ema_S = last_temperature;
  ema_MS = last_temperature * last_temperature;
  ema_M2 = last_temperature * last_temperature;
}

// Temperature read / simulate
float read_temperature() {
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);

  if (tempC == DEVICE_DISCONNECTED_C) {
    sensor_connected = false;
    return last_temperature;
  } else {
    if (!sensor_connected) {
      sensor_connected = true;
      for (int i = 0; i < SEQ_LEN; i++) temperature_history[i] = tempC;
      history_index = 0; history_count = SEQ_LEN;
      ema_M = tempC; ema_S = tempC; ema_MS = tempC*tempC; ema_M2 = tempC*tempC;
    }
    return tempC;
  }
}

void simulate_temperature_variation() {
  float drift = ((random(-100,101)/1000.0f) * 0.1f);
  float simulated = last_temperature + drift;
  temperature_history[history_index] = constrain(simulated, -50.0f, 125.0f);
  history_index = (history_index + 1) % SEQ_LEN;
  if (history_count < SEQ_LEN) history_count++;
}

// LSTM prediction
float make_prediction() {
  if (history_count < SEQ_LEN) return last_temperature;
  float raw_seq[SEQ_LEN], scaled_seq[SEQ_LEN];
  for (int i = 0; i < SEQ_LEN; i++) {
    int idx = (history_index + i) % SEQ_LEN;
    raw_seq[i] = temperature_history[idx];
  }
  for (int i = 0; i < SEQ_LEN; i++) scaled_seq[i] = scale_input(raw_seq[i]);
  return scale_output(predict_temperature(scaled_seq));
}

