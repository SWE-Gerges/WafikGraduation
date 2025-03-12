from django.urls import path, include

urlpatterns = [
    path("mobileApi/", include("mobileApi.urls")),
]