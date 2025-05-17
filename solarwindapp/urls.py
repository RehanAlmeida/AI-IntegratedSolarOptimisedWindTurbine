from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("modelpredication/", views.model_prediction, name="modelpredication"),
    path("resultspage/", views.results_page, name="resultspage"),
    path("aboutus/", views.about_us, name="aboutus"),
    path("userlogin/", views.user_login, name="userlogin"),
    path("userregister/", views.user_register, name="userregister"),
    path("userlogout/", views.user_logout, name="userlogout"),

    path("uploadsensordata/", views.upload_sensor_data, name="upload_sensor_data"),

]
