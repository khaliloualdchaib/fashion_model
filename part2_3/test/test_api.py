import requests

BASE_URL = "http://127.0.0.1:8003"

def test_recommendations():
    body_types = ["hourglass", "apple", "rectangle", "invalid"]

    for body_type in body_types:
        response = requests.get(f"{BASE_URL}/recommend?bodytype={body_type}")
        print(f"Testing for body type: {body_type}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}\n")

if __name__ == "__main__":
    test_recommendations()
