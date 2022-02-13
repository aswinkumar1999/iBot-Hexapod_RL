#include <WiFi.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#define RXD2 16
#define TXD2 17
Adafruit_MPU6050 mpu;

const char* ssid = "Cadbury";
const char* password = "chocolateS";
 
WiFiServer wifiServer(80);

void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit MPU6050 test!");

  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  Serial.println("");
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);
  WiFi.begin(ssid, password);
  delay(1000);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi..");
  }
 
  Serial.println("Connected to the WiFi network");
  Serial.println(WiFi.localIP());
 
  wifiServer.begin();
  delay(100);
}

void loop() {
 
  WiFiClient client = wifiServer.available();
 
  if (client) {
 
    while (client.connected()) {
 
      while (client.available()>0) {
        Serial2.print(char(client.read()));
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        
        Serial.print("Rotation X: ");
        Serial.print(g.gyro.x);
        client.write(g.gyro.x);
        Serial.print(", Y: ");
        Serial.print(g.gyro.y);
        Serial.print(", Z: ");
        Serial.print(g.gyro.z);
        Serial.println(" rad/s");
        
        Serial.println("");
        delay(50);
      }

    delay(50);
    /* Get new sensor events with the readings */
 
    
    }
 
    client.stop();
    Serial.println("Client disconnected");
 
  }
}
