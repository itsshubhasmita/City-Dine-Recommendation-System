import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# Load the corrected CSV data into a DataFrame
df = pd.read_csv("extendeddataset.csv")

# Encode categorical features
label_encoders = {}
for column in ['location', 'cuisine', 'budget', 'city', 'restaurants']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split features and target variable
X = df[['location', 'cuisine', 'budget', 'city']]
y_rating = df['rating']

# Train a decision tree regression model
rating_model = DecisionTreeRegressor()
rating_model.fit(X, y_rating)

def predict_restaurant(location, cuisine, budget, city):
    # Transform user input using the label encoders
    location_encoded = label_encoders['location'].transform([location])[0]
    cuisine_encoded = label_encoders['cuisine'].transform([cuisine])[0]
    budget_encoded = label_encoders['budget'].transform([budget])[0]
    city_encoded = label_encoders['city'].transform([city])[0]
    
    # Predict the rating for the given input
    predicted_rating = rating_model.predict([[location_encoded, cuisine_encoded, budget_encoded, city_encoded]])
    
    # Inverse transform to get the restaurant name
    restaurant_name = label_encoders['restaurants'].inverse_transform([int(predicted_rating)])[0]
    
    return restaurant_name, predicted_rating[0]

# Take user input
location = input("Enter location: ")
cuisine = input("Enter cuisine: ")
budget = input("Enter budget: ")
city = input("Enter city: ")

# Predict restaurant and rating
predicted_restaurant, predicted_rating = predict_restaurant(location, cuisine, budget, city)

# Display the prediction
print("Predicted restaurant:", predicted_restaurant)
print("Predicted rating:", predicted_rating)