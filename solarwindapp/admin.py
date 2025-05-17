from django.contrib import admin

from .models import SensorData, Prediction

# # Register your models here.

admin.site.register(SensorData)
admin.site.register(Prediction)
