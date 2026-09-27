// Relais USB <-> radar pour le kit Seeed XIAO MR60BHA2 (XIAO ESP32C6).
//
// Le firmware ESPHome d'origine envoie les mesures en Wi-Fi (Home Assistant),
// pas sur l'USB.  Ce croquis recopie octet pour octet les trames du radar
// (UART0 du XIAO, 115200 bauds, protocole TinyFrame) vers le port USB : le
// PC les décode avec radar/mmwave.py (radar run --mr60 COMx).
//
// Arduino IDE : carte « XIAO_ESP32C6 » (paquet esp32 d'Espressif),
// Outils → « USB CDC On Boot : Enabled », puis Téléverser.
// Pour revenir au firmware d'origine : wiki Seeed « Getting started with MR60BHA2 ».
//
// Même principe que l'exemple ReadByte de la bibliothèque Seeed_Arduino_mmWave,
// mais en binaire brut (pas de conversion en texte hexadécimal).

#include <Arduino.h>
#include <HardwareSerial.h>

HardwareSerial mmwaveSerial(0);   // UART0 : relié au module MR60BHA2 sur le kit

void setup() {
  Serial.begin(115200);           // USB (CDC) vers le PC
  mmwaveSerial.begin(115200);     // radar : 115200 8N1
}

void loop() {
  uint8_t buf[256];
  int n = mmwaveSerial.available();
  if (n > 0) {
    n = mmwaveSerial.readBytes(buf, min(n, (int)sizeof(buf)));
    Serial.write(buf, n);
  }
  // sens inverse (commandes éventuelles du PC vers le radar)
  while (Serial.available()) mmwaveSerial.write(Serial.read());
}
