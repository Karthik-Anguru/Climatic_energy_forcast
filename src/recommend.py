

def get_recommendation(prediction):
    if prediction > 4.0:
        return "⚠️ High usage – Try using heavy appliances during off-peak hours, or check insulation."
    elif prediction > 2.0:
        return "🔄 Medium usage – Consider switching off unused electronics and using LED lights."
    else:
        return "✅ Great! Your energy usage is efficient."
