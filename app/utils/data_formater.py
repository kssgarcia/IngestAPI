def formatter(user_data):
    activity_names = []
    for daily_activity in user_data.dietetics.daily_activities:
        for activity in daily_activity.activities:
            activity_names.append(activity.name)

    print(set(activity_names))

    formatted_data = (
            f"name:{user_data.name},"
            f"lastName:{user_data.lastName},"
            f"Email: {user_data.email}, "
            f"Age: {user_data.dietetics.age}, "
            f"Lifestyle: {user_data.dietetics.lifestyle}, "
            f"Job: {user_data.dietetics.job}, "
            f"Anthropometry: (Height: {user_data.dietetics.anthropometry.height}, "
            f"Weight: {user_data.dietetics.anthropometry.weight}, "
            f"BMI: {user_data.dietetics.anthropometry.BMI}, "
            f"Waist Circumference: {user_data.dietetics.anthropometry.waist_circumference}), "
            f"Biochemical Indicators: (Glucose: {user_data.dietetics.biochemical_indicators.glucose}, "
            f"Cholesterol: {user_data.dietetics.biochemical_indicators.cholesterol}), "
            f"Diet: (Ingest Preferences: {', '.join(user_data.dietetics.diet.ingest_preferences)}, "
            f"Fruits and Vegetables: {user_data.dietetics.diet.fruits_and_vegetables}, "
            f"Fiber: {user_data.dietetics.diet.fiber}, "
            f"Saturated Fats: {user_data.dietetics.diet.saturated_fats}, "
            f"Sugars: {user_data.dietetics.diet.sugars}, "
            f"Today Meals: {', '.join(user_data.dietetics.diet.today_meals)}), "
            f"Social Indicators: (Marital Status: {user_data.dietetics.social_indicators.marital_status}, "
            f"Income: {user_data.dietetics.social_indicators.income}, "
            f"Access to Healthy Foods: {user_data.dietetics.social_indicators.access_to_healthy_foods}), "
            f"Physical Activity: {user_data.dietetics.physical_activity}, "
            f"Daily Activities: {', '.join(activity_names)}, "
            f"Activities Energy Consumption: {user_data.dietetics.activities_energy_consumption}, "
            f"Goals: (Reduce Weight: {user_data.dietetics.goals}, "
            f"Potential Diseases: (Type 2 Diabetes: {user_data.dietetics.potential_diseases}, "
        )
    return formatted_data