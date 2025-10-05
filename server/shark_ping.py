import json
from datetime import datetime,timezone

data = {
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "shark_id": "SHARK_001",
    "jaw_motion": {"angular_velocity_deg_per_s": 12.5},
    "water_acceleration": {"m_per_s2": 3.8},
    "location": {"latitude": -33.857, "longitude": 151.215},
    "light_level": {"lux": 120, "period": "day"},
    "depth": {"meters": 45.2}
}

with open("shark_ping.json", "w") as f:
    json.dump(data, f, indent=4)

    