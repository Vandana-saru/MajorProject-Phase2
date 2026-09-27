/*
 * PRAHARI — ESP32 sensor node
 * Tier 1: acquire PPG at exactly 100 Hz, window it, stream to the Pi hub.
 *
 * Board:  ESP32 Dev Module (Arduino core for ESP32)
 * Serial: 921600 baud
 *
 * ---------------------------------------------------------------------------
 * PICK YOUR SENSOR — set SENSOR_MODE below before flashing.
 *
 *   SENSOR_ANALOG    any analog pulse sensor (the round "Pulse Sensor",
 *                    TCRT5000 module, most cheap lab-stock PPG boards)
 *
 *                    WIRING
 *                      sensor VCC  ->  ESP32 3V3      (NOT 5V — the ADC is
 *                                                      3.3V tolerant only)
 *                      sensor GND  ->  ESP32 GND
 *                      sensor OUT  ->  ESP32 GPIO34
 *
 *                    GPIO34 is on ADC1. Do not move this to an ADC2 pin
 *                    (GPIO0/2/4/12-15/25-27) — ADC2 stops working the moment
 *                    Wi-Fi is enabled, and it fails silently.
 *
 *   SENSOR_MAX30102  MAX30102 / MAX30105 breakout over I2C
 *
 *                    WIRING
 *                      VIN -> 3V3      SCL -> GPIO22
 *                      GND -> GND      SDA -> GPIO21
 *
 *                    Needs the SparkFun MAX3010x library:
 *                      Library Manager -> "SparkFun MAX3010x"
 * ---------------------------------------------------------------------------
 *
 * OUTPUT FORMAT (contracts.py :: FRAME_REQUIRED_KEYS)
 *   One JSON line per 5-second window, 50% overlap:
 *     {"seq":41,"t_ms":205300,"fs":100,"n":500,"jit_us":12,"samples":[...]}
 *
 *   Plus, once per second, a human-readable status line prefixed with '#'
 *   which the hub ignores:
 *     # rate=100.0Hz jitter_us=11 raw=48213 free_heap=212344
 */

// =============================================================== CONFIGURATION
#define SENSOR_ANALOG     1
#define SENSOR_MAX30102   2

#define SENSOR_MODE       SENSOR_ANALOG      // <<<<<< SET THIS

#define ANALOG_PIN        34                 // ADC1 only
#define SAMPLE_RATE_HZ    100
#define WINDOW_SAMPLES    500                // 5.0 s  — must match contracts.py
#define HOP_SAMPLES       250                // 50% overlap
#define SERIAL_BAUD       921600

// ====================================================================== INCLUDES
#if SENSOR_MODE == SENSOR_MAX30102
  #include <Wire.h>
  #include "MAX30105.h"
  MAX30105 sensor;
#endif

// ======================================================================= STATE
static const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

volatile uint32_t isrTicks = 0;
hw_timer_t *sampleTimer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// Ring buffer holding one window plus one hop, so a full window is always
// available while the next hop fills.
static float   ring[WINDOW_SAMPLES];
static uint16_t ringHead   = 0;
static uint32_t totalSamples = 0;
static uint32_t seq = 0;

// jitter tracking — this is WP1's sampling-validation evidence
static uint32_t lastSampleUs = 0;
static int32_t  jitterMaxUs  = 0;
static uint32_t statusLastMs = 0;
static float    lastRaw      = 0;

void IRAM_ATTR onSampleTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  isrTicks++;
  portEXIT_CRITICAL_ISR(&timerMux);
}

// ======================================================================== SETUP
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);
  Serial.println("# PRAHARI ESP32 node booting");

#if SENSOR_MODE == SENSOR_ANALOG
  analogReadResolution(12);                 // 0..4095
  // 11 dB attenuation gives the full 0-3.3 V input range
  analogSetPinAttenuation(ANALOG_PIN, ADC_11db);
  Serial.println("# sensor: ANALOG on GPIO" + String(ANALOG_PIN) + " (ADC1)");
#else
  Wire.begin(21, 22);
  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("# FATAL: MAX3010x not found. Check wiring and pull-ups.");
    while (1) delay(1000);
  }
  // 100 Hz at the sensor so no resampling is needed on the hub.
  //   ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange
  sensor.setup(0x1F, 4, 2, 400, 411, 4096);   // 400 Hz / avg 4 => 100 Hz out
  sensor.setPulseAmplitudeRed(0x0A);
  sensor.setPulseAmplitudeIR(0x1F);
  Serial.println("# sensor: MAX30102 over I2C, 100 Hz effective");
#endif

  // Hardware timer: 80 MHz / 80 = 1 MHz tick, fire every SAMPLE_PERIOD_US
  sampleTimer = timerBegin(0, 80, true);
  timerAttachInterrupt(sampleTimer, &onSampleTimer, true);
  timerAlarmWrite(sampleTimer, SAMPLE_PERIOD_US, true);
  timerAlarmEnable(sampleTimer);

  lastSampleUs = micros();
  Serial.println("# streaming at " + String(SAMPLE_RATE_HZ) + " Hz");
}

// ================================================================ SAMPLE READ
static inline float readSensor() {
#if SENSOR_MODE == SENSOR_ANALOG
  return (float) analogRead(ANALOG_PIN);
#else
  // IR channel is the better PPG source on the MAX30102
  return (float) sensor.getIR();
#endif
}

// ============================================================== FRAME EMITTER
void emitWindow() {
  uint32_t now = millis();
  Serial.print("{\"seq\":");     Serial.print(seq);
  Serial.print(",\"t_ms\":");    Serial.print(now);
  Serial.print(",\"fs\":");      Serial.print(SAMPLE_RATE_HZ);
  Serial.print(",\"n\":");       Serial.print(WINDOW_SAMPLES);
  Serial.print(",\"jit_us\":");  Serial.print(jitterMaxUs);
  Serial.print(",\"samples\":[");

  // oldest sample first
  for (uint16_t i = 0; i < WINDOW_SAMPLES; i++) {
    uint16_t idx = (ringHead + i) % WINDOW_SAMPLES;
    Serial.print(ring[idx], 2);
    if (i < WINDOW_SAMPLES - 1) Serial.print(',');
  }
  Serial.println("]}");

  seq++;
  jitterMaxUs = 0;
}

// ========================================================================= LOOP
void loop() {
  uint32_t ticks;
  portENTER_CRITICAL(&timerMux);
  ticks = isrTicks;
  isrTicks = 0;
  portEXIT_CRITICAL(&timerMux);

  while (ticks--) {
    uint32_t nowUs = micros();
    int32_t  dt    = (int32_t)(nowUs - lastSampleUs);
    int32_t  err   = dt - (int32_t)SAMPLE_PERIOD_US;
    if (abs(err) > abs(jitterMaxUs)) jitterMaxUs = err;
    lastSampleUs = nowUs;

    float v = readSensor();
    lastRaw = v;
    ring[ringHead] = v;
    ringHead = (ringHead + 1) % WINDOW_SAMPLES;
    totalSamples++;

    // emit once a full window exists, then every HOP samples
    if (totalSamples >= WINDOW_SAMPLES &&
        ((totalSamples - WINDOW_SAMPLES) % HOP_SAMPLES) == 0) {
      emitWindow();
    }
  }

  // status line once per second
  uint32_t ms = millis();
  if (ms - statusLastMs >= 1000) {
    statusLastMs = ms;
    Serial.print("# rate_target="); Serial.print(SAMPLE_RATE_HZ);
    Serial.print("Hz jitter_us=");  Serial.print(jitterMaxUs);
    Serial.print(" raw=");          Serial.print(lastRaw, 1);
    Serial.print(" total=");        Serial.print(totalSamples);
    Serial.print(" free_heap=");    Serial.println(ESP.getFreeHeap());
  }
}
