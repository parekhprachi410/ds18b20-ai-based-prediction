#include "temp_predictor.h"
#include "lstm_inference.h"
#include "scaler.h"

#include <Arduino.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BluetoothSerial.h>

#ifndef SEQ_LEN
#define SEQ_LEN 60
#endif

#define ONE_WIRE_BUS 4

// Prediction interval kept small to observe behavior clearly during testing
const unsigned long PREDICTION_INTERVAL = 6000UL;

// Blending and correction parameters tuned experimentally
const float BLEND_BETA = 0.65f;
const float BIAS_ALPHA = 0.02f;
const float PARAM_ALPHA = 0.3f;
const float SLOPE_MIN = 0.7f;
const float SLOPE_MAX = 1.3f;
const float EPS_NUM = 1e-6f;

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
BluetoothSerial BT;

// Circular buffer to store past temperature values for LSTM input
float temperature_history[SEQ_LEN];
int history_index = 0;
int history_count = 0;

bool sensor_connected = true;
bool bt_connected = false;

// Used as fallback during temporary sensor failures
float last_temperature = 22.0f;
unsigned long last_prediction = 0;

// Variables for online linear bias correction
float ema_M = 0, ema_S = 0, ema_MS = 0, ema_M2 = 0;
float learned_slope = 1.0f;
float learned_intercept = 0.0f;
bool linear_initialized = false;

// Bluetooth status callbacks help decide where output should go
void onBTConnect(esp_spp_cb_event_t, esp_spp_cb_param_t *)
{
  bt_connected = true;
  Serial.println("Bluetooth connected");
}

void onBTDisconnect(esp_spp_cb_event_t, esp_spp_cb_param_t *)
{
  bt_connected = false;
  Serial.println("Bluetooth disconnected");
}

float read_temperature();
float make_prediction();
void simulate_temperature_variation();
void update_linear_correction(float model_pred, float measured);
float apply_linear_correction(float model_pred);
void reset_linear_params();
void checkSensorStatus();

void setup()
{
  Serial.begin(115200);

  BT.begin("ESP32_BT");
  BT.register_callback([](esp_spp_cb_event_t event, esp_spp_cb_param_t *param) {
    if (event == ESP_SPP_SRV_OPEN_EVT) onBTConnect(event, param);
    if (event == ESP_SPP_CLOSE_EVT) onBTDisconnect(event, param);
  });

  sensors.begin();
  sensors.setResolution(12);
  sensor_connected = sensors.getDeviceCount() > 0;

  // Initialize LSTM internal states
  lstm_init();

  // Pre-fill history so the model has valid input immediately
  float initTemp = read_temperature();
  for (int i = 0; i < SEQ_LEN; i++)
    temperature_history[i] = initTemp;

  history_count = SEQ_LEN;
  last_temperature = initTemp;

  reset_linear_params();
}

void loop()
{
  // Sensor status is checked continuously to allow quick recovery
  checkSensorStatus();

  if (millis() - last_prediction < PREDICTION_INTERVAL)
    return;

  last_prediction = millis();

  float current_temperature = read_temperature();
  float final_prediction;

  if (sensor_connected)
  {
    // When sensor is available, prediction is gently aligned with real data
    float model_pred = make_prediction();
    update_linear_correction(model_pred, current_temperature);

    final_prediction =
      BLEND_BETA * apply_linear_correction(model_pred) +
      (1.0f - BLEND_BETA) * current_temperature;
  }
  else
  {
    // During sensor failure, prediction relies on past trends
    simulate_temperature_variation();
    float model_pred = make_prediction();

    final_prediction =
      BLEND_BETA * apply_linear_correction(model_pred) +
      (1.0f - BLEND_BETA) * last_temperature;
  }

  last_temperature = current_temperature;

  String output;

  // Output format is kept human-readable for easy monitoring
  if (sensor_connected)
  {
    output  = "Connected:-\n";
    output += "Real-time temperature = ";
    output += String(current_temperature, 2);
    output += " °C\n\n";
  }
  else
  {
    output  = "Disconnected:-\n";
    output += "Predicted temperature = ";
    output += String(final_prediction, 2);
    output += " °C\n\n";
  }

  if (bt_connected)
    BT.print(output);
  else
    Serial.print(output);
}

void checkSensorStatus()
{
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);

  if (tempC != DEVICE_DISCONNECTED_C)
  {
    // On reconnection, buffers and correction parameters are reset
    if (!sensor_connected)
    {
      sensor_connected = true;

      for (int i = 0; i < SEQ_LEN; i++)
        temperature_history[i] = tempC;

      history_index = 0;
      history_count = SEQ_LEN;
      reset_linear_params();

      Serial.println("Sensor reconnected - real-time mode restored");
      if (bt_connected)
        BT.println("Sensor reconnected - real-time mode restored");
    }
  }
  else
  {
    sensor_connected = false;
  }
}

void update_linear_correction(float model_pred, float measured)
{
  // Initial values are set on first valid comparison
  if (!linear_initialized)
  {
    ema_M = model_pred;
    ema_S = measured;
    ema_MS = model_pred * measured;
    ema_M2 = model_pred * model_pred;
    linear_initialized = true;
    return;
  }

  // EMA-based update allows slow adaptation without instability
  ema_M  = (1 - BIAS_ALPHA) * ema_M  + BIAS_ALPHA * model_pred;
  ema_S  = (1 - BIAS_ALPHA) * ema_S  + BIAS_ALPHA * measured;
  ema_MS = (1 - BIAS_ALPHA) * ema_MS + BIAS_ALPHA * model_pred * measured;
  ema_M2 = (1 - BIAS_ALPHA) * ema_M2 + BIAS_ALPHA * model_pred * model_pred;

  float denom = ema_M2 - ema_M * ema_M;
  float slope = fabs(denom) > EPS_NUM ?
                (ema_MS - ema_M * ema_S) / denom : 1.0f;

  slope = constrain(slope, SLOPE_MIN, SLOPE_MAX);

  learned_slope =
    (1 - PARAM_ALPHA) * learned_slope + PARAM_ALPHA * slope;

  learned_intercept = ema_S - learned_slope * ema_M;
}

float apply_linear_correction(float model_pred)
{
  return learned_slope * model_pred + learned_intercept;
}

void reset_linear_params()
{
  linear_initialized = false;
  learned_slope = 1.0f;
  learned_intercept = 0.0f;
}

float read_temperature()
{
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);

  if (tempC == DEVICE_DISCONNECTED_C)
  {
    sensor_connected = false;
    return last_temperature;
  }

  temperature_history[history_index] = tempC;
  history_index = (history_index + 1) % SEQ_LEN;
  if (history_count < SEQ_LEN)
    history_count++;

  return tempC;
}

void simulate_temperature_variation()
{
  // Small random drift prevents the sequence from becoming static
  float drift = random(-10, 10) / 100.0f;
  temperature_history[history_index] = last_temperature + drift;
  history_index = (history_index + 1) % SEQ_LEN;
}

float make_prediction()
{
  if (history_count < SEQ_LEN)
    return last_temperature;

  float raw[SEQ_LEN];
  float scaled[SEQ_LEN];

  for (int i = 0; i < SEQ_LEN; i++)
  {
    raw[i] = temperature_history[(history_index + i) % SEQ_LEN];
    scaled[i] = scale_input(raw[i]);
  }

  return scale_output(predict_temperature(scaled));
}
