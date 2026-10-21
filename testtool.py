import requests

def get_weather(city):
    """
    Get current weather information for a city without an API key.
    """

    try:
        url = f"https://wttr.in/{city}"

        params = {
            "format": "j1"
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        current = data["current_condition"][0]

        weather_info = {
            "city": city,
            "temperature": f"{current['temp_C']} °C",
            "feels_like": f"{current['FeelsLikeC']} °C",
            "condition": current["weatherDesc"][0]["value"],
            "humidity": f"{current['humidity']}%",
            "wind_speed": f"{current['windspeedKmph']} km/h"
        }

        return weather_info

    except Exception as e:
        return {
            "error": f"Unable to get weather: {str(e)}"
        }


# --------------------------
# Function Call
# --------------------------

city = input("Enter city name: ")

weather = get_weather(city)

print("\nWeather Information")
print("-------------------")

if "error" in weather:
    print(weather["error"])
else:
    print("City:", weather["city"])
    print("Temperature:", weather["temperature"])
    print("Feels Like:", weather["feels_like"])
    print("Condition:", weather["condition"])
    print("Humidity:", weather["humidity"])
    print("Wind Speed:", weather["wind_speed"])