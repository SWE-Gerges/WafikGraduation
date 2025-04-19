from django.urls import path
from .views import predict, home

urlpatterns = [
    path('', home, name='home'),
    path("predict/", predict, name="predict"),
]