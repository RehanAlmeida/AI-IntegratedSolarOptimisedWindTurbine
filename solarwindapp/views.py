from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .models import SensorData, Prediction
import joblib
import pandas as pd

from django.http import JsonResponse
from django.utils.timezone import now
import uuid

import google.generativeai as genai


# Configure Gemini API Key
genai.configure(api_key="AIzaSyDXX5WRHlWICkoQ71pcz4wnrd8TKbn-D-Y")

def generate_ai_response(prompt: str, model_name: str = "gemini-1.5-flash"):
    """
    Sends a prompt to Gemini AI and returns the generated response.

    :param prompt: str - The prompt to send.
    :param model_name: str - The Gemini model to use.
    :return: str - AI-generated response.
    """
    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt)
        return response.text if response else None
    except Exception as e:
        print(f"[ERROR] Failed to generate AI response: {e}")
        return None


# Load the trained model
MODEL_PATH = "solar_wind_classifier.pkl"
model = joblib.load(MODEL_PATH)
FEATURES = ['wind_speed', 'humidity', 'temperature', 'wind_direction']

def upload_sensor_data(request):
    """
    API Endpoint: /uploadsensordata?nodeid=123&dcv=10&dci=230&acv=240

    - Retrieves sensor data from query parameters.
    - Stores the data in the database.
    - Returns a JSON response with success or error messages.
    """
    try:
        # Extracting parameters
        nodeid = request.GET.get("nodeid")
        dc_voltage = request.GET.get("dcv")
        dc_current = request.GET.get("dci")
        ac_voltage = request.GET.get("acv")

        # Validate required parameters
        if not all([nodeid, dc_voltage, dc_current, ac_voltage]):
            return JsonResponse({"status": "error", "message": "Missing parameters"}, status=400)

        # Create and save sensor data entry
        sensor_entry = SensorData.objects.create(
            nodeid=nodeid,
            dcvoltage=dc_voltage,
            dccurrent=dc_current,
            acvoltage=ac_voltage,
            timestamp=now()
        )

        return JsonResponse({
            "status": "success",
            "message": "Sensor data uploaded successfully",
            "data": {
                "nodeid": sensor_entry.nodeid,
                "dcvoltage": sensor_entry.dcvoltage,
                "dccurrent": sensor_entry.dccurrent,
                "acvoltage": sensor_entry.acvoltage,
                "timestamp": sensor_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Failed to upload data: {str(e)}"}, status=500)


# @login_required(login_url='/userlogin')
# def dashboard(request):
#     try:
#         sensor_data = SensorData.objects.order_by('-timestamp')[:20]
#     except Exception as e:
#         sensor_data = []

#     return render(request, "dashboard.html", {"sensor_data": sensor_data})


@login_required(login_url='/userlogin')
def dashboard(request):
    try:
        # Retrieve the latest 20 sensor data records
        sensor_data = SensorData.objects.order_by('-timestamp')[:20]

        # Convert dcvoltage to float for each data record, handling invalid values
        for data in sensor_data:
            try:
                data.dcvoltage = float(data.dcvoltage)  # Try converting dcvoltage to float
            except (ValueError, TypeError):
                data.dcvoltage = None  # Set to None if conversion fails

    except Exception as e:
        sensor_data = []

    return render(request, "dashboard.html", {"sensor_data": sensor_data})




@login_required(login_url='/userlogin')
def model_prediction(request):
    if request.method == "POST":
        wind_speed = float(request.POST["wind_speed"])
        humidity = int(request.POST["humidity"])
        temperature = int(request.POST["temperature"])
        wind_direction = int(request.POST["wind_direction"])

        df_input = pd.DataFrame([[wind_speed, humidity, temperature, wind_direction]], columns=FEATURES)
        pred = model.predict(df_input)[0]
        confidence = max(model.predict_proba(df_input)[0])

        prediction = Prediction.objects.create(
            title="Solar/Wind Prediction",
            wind_speed=wind_speed,
            humidity=humidity,
            temperature=temperature,
            wind_direction=wind_direction,
            prediction="Wind" if pred == 1 else "Solar"
        )

        return redirect(f"/resultspage?mid={prediction.mid}")

    return render(request, "modelpredication.html")

@login_required(login_url='/userlogin')
def results_page(request):
    mid = request.GET.get("mid")
    if mid:
        prediction = Prediction.objects.filter(mid=mid).first()
        return render(request, "resultspage.html", {"prediction": prediction})
    else:
        all_predictions = Prediction.objects.all()
        return render(request, "resultspage.html", {"predictions": all_predictions})

def about_us(request):

    return render(request, "aboutus.html")


def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("/")
    return render(request, "userlogin.html")

def user_register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        if User.objects.filter(username=username).exists():
            return render(request, "userregister.html", {"error": "Username already exists!"})
        User.objects.create_user(username=username, email=email, password=password)
        return redirect("/userlogin")
    return render(request, "userregister.html")

def user_logout(request):
    logout(request)
    return redirect("/userlogin")
