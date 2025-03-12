import joblib
import numpy as np
from django.http import JsonResponse
from rest_framework.decorators import api_view



# Model
model = joblib.load("linear_regression_model.pkl")

@api_view(["POST", "GET"])
def predict(request):
    try:
        data = request.data.get("features")  
        if not data:
            return JsonResponse({"error": "No input data provided"}, status=400)

        features = np.array(data).reshape(1, -1)
        prediction = model.predict(features)

        return JsonResponse({"prediction": prediction.tolist()})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
