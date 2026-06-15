
```python
from secure_rover_autopilot import run_autopilot

run_autopilot()
```

```powershell
$env:ROVER_IP = "192.168.4.1"
$env:PHONE_IP = "192.168.4.2"
$env:CAMERA_MODE = "IP_WEBCAM"
$env:TARGET_COLOR = "GREEN"
$env:PROCESS_FPS = "4.0"
$env:ROVER_HMAC_SECRET = "change-this-long-random-secret"
$env:ROVER_SECURITY_LOG = "rover_security.log"
```

```json
{
  "L": 0.2,
  "R": 0.2,
  "T": 1,
  "nonce": "random_hex",
  "seq": 42,
  "sig": "hmac_sha256_hex",
  "ts": 1710000000
}
```