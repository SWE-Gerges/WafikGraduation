import joblib
import os
from django.conf import settings
import numpy as np
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.shortcuts import render


model_path = os.path.join(settings.BASE_DIR, 'xgboost_real_estate_BestModel.pkl')
best_model = joblib.load(model_path)
FEATURE_ORDER = [
    "type_encoded", "location_encoded", "bedroom", "bathroom", "size_sqm",
    "price_per_sqm", "room_density", "bed_bath_ratio", "location_complexity", "size_category"
]

TYPE_MAPPING = {
    "Duplex": 5, "Villa": 8, "Apartment": 3, "Townhouse": 6, "Penthouse": 7,
    "iVilla": 6, "Twin House": 5, "Hotel Apartment": 4, "Chalet": 4, "Compound": 7
}

@api_view(["GET"])
# def predict(request):
#     try:
#         data = request.query_params
#         if not data:
#             return JsonResponse({"error": "No input data provided"}, status=400)

#         property_type = data.get("type")
#         location = data.get("location")
#         bedrooms = data.get("bedrooms")
#         bathrooms = data.get("bathrooms")
#         size_sqm = data.get("size_sqm")

#         if None in [property_type, location, bedrooms, bathrooms, size_sqm]:
#             return JsonResponse({"error": "Missing required fields"}, status=400)

#         try:
#             bedrooms = int(bedrooms)
#             bathrooms = int(bathrooms)
#             size_sqm = float(size_sqm)
#         except ValueError:
#             return JsonResponse({"error": "Invalid numeric input"}, status=400)

#         type_encoded = TYPE_MAPPING.get(property_type, 3)
#         location_encoded = -1  # You can update this later

#         room_density = bedrooms / size_sqm
#         bed_bath_ratio = bedrooms / (bathrooms + 1)
#         location_complexity = location.count(",") + 1
#         size_category = int(pd.cut([size_sqm], bins=[0, 100, 200, 300, np.inf], labels=[1, 2, 3, 4])[0])

#         input_data = pd.DataFrame([[ 
#             type_encoded, location_encoded, bedrooms, bathrooms, size_sqm,
#             0,
#             room_density, bed_bath_ratio, location_complexity, size_category
#         ]], columns=FEATURE_ORDER)

#         predicted_price = float(best_model.predict(input_data)[0])
#         return JsonResponse({"predicted_price": round(predicted_price, 2)})

#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)

def predict(request):
    try:
        data = request.query_params
        if not data:
            return JsonResponse({"error": "No input data provided"}, status=400)

        property_type = data.get("type")
        location = data.get("location")
        bedrooms = data.get("bedrooms")
        bathrooms = data.get("bathrooms")
        size_sqm = data.get("size_sqm")

        if None in [property_type, location, bedrooms, bathrooms, size_sqm]:
            return JsonResponse({"error": "Missing required fields"}, status=400)

        try:
            bedrooms = int(bedrooms)
            bathrooms = int(bathrooms)
            size_sqm = float(size_sqm)
        except ValueError:
            return JsonResponse({"error": "Invalid numeric input"}, status=400)

        # Create all required features
        type_encoded = TYPE_MAPPING.get(property_type, 3)
        location_encoded = -1  # Default value for unknown locations
        room_density = bedrooms / size_sqm
        bed_bath_ratio = bedrooms / (bathrooms + 1)
        location_complexity = location.count(",") + 1
        size_category = int(pd.cut([size_sqm], bins=[0, 100, 200, 300, np.inf], labels=[1, 2, 3, 4])[0])

        # Create a dictionary of input features
        input_dict = {
            'type_encoded': type_encoded,
            'location_encoded': location_encoded,
            'bedroom': bedrooms,
            'bathroom': bathrooms,
            'size_sqm': size_sqm,
            'room_density': room_density,
            'bed_bath_ratio': bed_bath_ratio,
            'location_complexity': location_complexity,
            'size_category': size_category
        }

        # Create DataFrame with all required columns
        input_data = pd.DataFrame([input_dict])
        
        # Ensure columns match exactly what the model expects
        for col in FEATURE_ORDER:
            if col not in input_data.columns:
                input_data[col] = 0  # Add missing columns with default value
        
        # Make sure columns are in the exact order the model expects
        input_data = input_data[FEATURE_ORDER]
        
        # # Debug information to troubleshoot column mismatch
        # print(f"Model expects columns: {FEATURE_ORDER}")
        # print(f"Input data has columns: {input_data.columns.tolist()}")
        # print(f"Input data has shape: {input_data.shape}")

        predicted_price = float(best_model.predict(input_data)[0])
        return JsonResponse({
            "predicted_price": round(predicted_price, 2),
            "features_used": FEATURE_ORDER
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

def home(request):
    return render(request, 'home.html')
