#include <WiFi.h>

#define RXD2 16
#define TXD2 17
  
const char* ssid = "OnePlus";
const char* password = "qxej0808";
 
WiFiServer wifiServer(80);
 
void setup() {

 
  Serial.begin(115200); 
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
}
 
void loop() {
 
  WiFiClient client = wifiServer.available();
 
  if (client) {
 
    while (client.connected()) {
 
      while (client.available()>0) {
        Serial2.print(char(client.read()));
      }
 
      delay(10);
    }
 
    client.stop();
    Serial.println("Client disconnected");
 
  }
}
