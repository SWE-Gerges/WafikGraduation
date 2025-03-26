import joblib
import numpy as np
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view

# Load the trained model
best_model = joblib.load("xgboost_real_estate_BestModel.pkl")

# Define feature order (must match training)
FEATURE_ORDER = [
    "type_encoded", "location_encoded", "bedroom", "bathroom", "size_sqm",
    "price_per_sqm", "room_density", "bed_bath_ratio", "location_complexity", "size_category"
]

# Property type mapping
TYPE_MAPPING = {
    "Duplex": 5, "Villa": 8, "Apartment": 3, "Townhouse": 6, "Penthouse": 7,
    "iVilla": 6, "Twin House": 5, "Hotel Apartment": 4, "Chalet": 4, "Compound": 7
}

@api_view(["POST"])
def predict(request):
    try:
        # Extract JSON input
        data = request.data
        if not data:
            return JsonResponse({"error": "No input data provided"}, status=400)

        # Extract fields
        property_type = data.get("type")
        location = data.get("location")
        bedrooms = data.get("bedrooms")
        bathrooms = data.get("bathrooms")
        size_sqm = data.get("size_sqm")

        # Validate input
        if None in [property_type, location, bedrooms, bathrooms, size_sqm]:
            return JsonResponse({"error": "Missing required fields"}, status=400)

        # Encode property type
        type_encoded = TYPE_MAPPING.get(property_type, 3)  # Default to Apartment

        # Handle unseen locations
        location_encoded = -1  # Default for unseen locations

        # Compute additional features
        room_density = bedrooms / size_sqm
        bed_bath_ratio = bedrooms / (bathrooms + 1)
        location_complexity = location.count(",") + 1
        size_category = pd.cut([size_sqm], bins=[0, 100, 200, 300, np.inf], labels=[1, 2, 3, 4])[0]

        # **Fix: Dynamically handle missing columns**
        input_data = pd.DataFrame([[
            type_encoded, location_encoded, bedrooms, bathrooms, size_sqm,
            0,  # Placeholder for price_per_sqm
            room_density, bed_bath_ratio, location_complexity, size_category
        ]], columns=FEATURE_ORDER)

        # **Fix: Match training column order**
        missing_cols = set(FEATURE_ORDER) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0  # Assign default value

        input_data = input_data[FEATURE_ORDER]  # Ensure column order

        # Predict price & convert to float
        predicted_price = float(best_model.predict(input_data)[0])

        return JsonResponse({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
