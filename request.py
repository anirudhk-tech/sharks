import json
import time
import requests
from datetime import datetime, timezone

URL = "https://server.com/ping" # Replace with server URL
SHARK_ID = "SHARK_001"

while True:
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "shark_id": SHARK_ID,
        "jaw_motion": {"angular_velocity_deg_per_s": 12.5},
        "water_acceleration": {"m_per_s2": 3.8},
        "location": {"latitude": -33.857, "longitude": 151.215},
        "light_level": {"lux": 120, "period": "day"},
        "depth": {"meters": 45.2}
    }
    # Send to server
    try:
        response = requests.post(URL, json=data)
        print(f"[{data['timestamp']}] Sent ping — Status: {response.status_code}")
    except Exception as e:
        print(f"[{data['timestamp']}] Failed to send ping: {e}")

    # Wait 20 seconds before next ping
    time.sleep(20)