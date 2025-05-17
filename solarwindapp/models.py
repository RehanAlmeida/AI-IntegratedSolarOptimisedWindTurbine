from django.db import models
from django.contrib.auth.models import User

class SensorData(models.Model):
    nodeid = models.TextField()
    dcvoltage = models.TextField()
    dccurrent = models.TextField()
    acvoltage = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.nodeid} - {self.dcvoltage}V, {self.dccurrent}A, {self.acvoltage}V"

class Prediction(models.Model):
    mid = models.AutoField(primary_key=True)
    title = models.TextField()
    wind_speed = models.FloatField()
    humidity = models.IntegerField()
    temperature = models.IntegerField()
    wind_direction = models.IntegerField()
    prediction = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.mid} - {self.prediction}"
