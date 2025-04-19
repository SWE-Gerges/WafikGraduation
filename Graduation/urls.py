from django.urls import path, include
from mobileApi.views import home

urlpatterns = [
    path("mobileApi/", include("mobileApi.urls")),
    path('', home, name='home'),
]