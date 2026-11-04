
#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"

// 1. Define two MPU6050 sensors

MPU6050 mpu1(0x68);
MPU6050 mpu2(0x69);

// 2. Define raw data variables

int16_t ax1, ay1, az1;
int16_t gx1, gy1, gz1;

int16_t ax2, ay2, az2;
int16_t gx2, gy2, gz2;

// 3. Define sequence number and timestamp

unsigned long seq = 0;
unsigned long timestamp_ms = 0;

// 4. Sampling configuration

const unsigned long SAMPLE_INTERVAL_US = 20000;
unsigned long nextSampleTime = 0;

#define LED_PIN 13

bool blinkState = false;

// 5. Setup

void setup() {

  Wire.begin();

  Serial.begin(38400);

  Serial.println("Initializing two MPU6050 sensors...");

  // Initialize both sensors

  mpu1.initialize();
  mpu2.initialize();

  // Set accelerometer range: +/-4g

  mpu1.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);
  mpu2.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);

  // Set gyroscope range: +/-500 deg/s

  mpu1.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
  mpu2.setFullScaleGyroRange(MPU6050_GYRO_FS_500);

  // Check sensor connections

  Serial.println("Testing device connections...");

  bool sensor1Connected = mpu1.testConnection();
  bool sensor2Connected = mpu2.testConnection();

  if (sensor1Connected) {
    Serial.println("MPU6050 #1 connection successful");
  } else {
    Serial.println("MPU6050 #1 connection failed");
  }

  if (sensor2Connected) {
    Serial.println("MPU6050 #2 connection successful");
  } else {
    Serial.println("MPU6050 #2 connection failed");
  }

  // Configure LED

  pinMode(LED_PIN, OUTPUT);

  // Stop if either sensor is not connected

  if (!sensor1Connected || !sensor2Connected) {

    Serial.println("Please check the wiring.");

    while (true) {
      digitalWrite(LED_PIN, HIGH);
      delay(200);

      digitalWrite(LED_PIN, LOW);
      delay(200);
    }
  }

  // Print CSV header

  Serial.println(
    "seq,timestamp_ms,"
    "ax1,ay1,az1,gx1,gy1,gz1,"
    "ax2,ay2,az2,gx2,gy2,gz2");

  // Initialize sampling timer

  nextSampleTime = micros();
}

// 6. Main loop

void loop() {

  // Wait until the next sampling time

  unsigned long now = micros();

  if ((long)(now - nextSampleTime) < 0) {
    return;
  }

  // Schedule next sample

  nextSampleTime += SAMPLE_INTERVAL_US;

  // Record sampling timestamp

  timestamp_ms = millis();

  // Read MPU6050 #1

  mpu1.getAcceleration(&ax1, &ay1, &az1);
  mpu1.getRotation(&gx1, &gy1, &gz1);

  // Read MPU6050 #2

  mpu2.getAcceleration(&ax2, &ay2, &az2);
  mpu2.getRotation(&gx2, &gy2, &gz2);

  // Output sequence number and timestamp

  Serial.print(seq++);
  Serial.print(",");

  Serial.print(timestamp_ms);
  Serial.print(",");

  // Output MPU6050 #1 data

  Serial.print(ax1);
  Serial.print(",");

  Serial.print(ay1);
  Serial.print(",");

  Serial.print(az1);
  Serial.print(",");

  Serial.print(gx1);
  Serial.print(",");

  Serial.print(gy1);
  Serial.print(",");

  Serial.print(gz1);
  Serial.print(",");

  // Output MPU6050 #2 data

  Serial.print(ax2);
  Serial.print(",");

  Serial.print(ay2);
  Serial.print(",");

  Serial.print(az2);
  Serial.print(",");

  Serial.print(gx2);
  Serial.print(",");

  Serial.print(gy2);
  Serial.print(",");

  Serial.println(gz2);

  // Blink LED

  blinkState = !blinkState;
  digitalWrite(LED_PIN, blinkState);

  // Handle sampling overrun

  if ((long)(micros() - nextSampleTime) >= 0) {
    nextSampleTime = micros() + SAMPLE_INTERVAL_US;
  }
}
