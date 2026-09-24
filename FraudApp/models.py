from django.db import models

class Register(models.Model):
    username = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    contact = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    address = models.CharField(max_length=200)