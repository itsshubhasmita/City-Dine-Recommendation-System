import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("extendeddataset.csv")

# Encode categorical features
label_encoders = {}
for column in ['location', 'cuisine', 'budget', 'city', 'restaurants']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split features and target variable
X = df[['location', 'cuisine', 'budget', 'city']]
y_rating = df['rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_rating, test_size=0.2, random_state=42)

# Train a decision tree regression model
rating_model = DecisionTreeRegressor()
rating_model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = rating_model.predict(X_train)

# Make predictions on the testing set
y_test_pred = rating_model.predict(X_test)

# Evaluate the model's accuracy using mean squared error
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Training Mean Squared Error:", train_mse)
print("Testing Mean Squared Error:", test_mse)
