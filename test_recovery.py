import requests
import json

res = requests.get("http://localhost:8000/recover?loss=1000&currency=INR")
print(res.status_code)
print(json.dumps(res.json(), indent=2))
