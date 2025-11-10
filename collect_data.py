import serial
import time
import csv

# Replace 'COM5' with your ESP32's serial port
ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)  # Wait for serial connection to initialize

data = []
start_time = time.time()
duration = 7200  # Collect for 30 minutes (1800 seconds); adjust as needed

print("Collecting DS18B20 temperature data... Press Ctrl+C to stop early.\n")

try:
    while time.time() - start_time < duration:
        line = ser.readline().decode('utf-8').strip()
        if not line or line.lower().startswith("timestamp"):
            continue  # Skip empty lines or header
        parts = line.split(',')
        # Expecting format: timestamp,temperature
        if len(parts) == 2:
            try:
                timestamp = float(parts[0])
                temperature = float(parts[1])
                data.append([timestamp, temperature])
                print(f"Collected: {line}")
            except ValueError:
                print(f"Skipped invalid line: {line}")
except KeyboardInterrupt:
    print("\nStopped early by user.")

ser.close()

# Save to CSV
if data:
    filename = 'ds18b20_data.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'temperature_C'])
        writer.writerows(data)
    print(f"\n✅ Data saved to {filename} ({len(data)} rows).")
else:
    print("\n⚠️ No data collected. Check ESP32 connection and serial output.")
