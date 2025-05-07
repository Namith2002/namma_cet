import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the training data
print("Loading training data...")
train_data = pd.read_csv('processed_data/synthetic_training_data.csv')

# Display basic information about the dataset
print(f"Dataset shape: {train_data.shape}")
print("\nFirst few rows of the dataset:")
print(train_data.head())

# Define features and target
X = train_data[['physics_kcet', 'chemistry_kcet', 'mathematics_kcet', 'biology_kcet', 
                'physics_theory', 'chemistry_theory', 'mathematics_theory', 'biology_theory']]
y = train_data['rank']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train multiple models and evaluate their performance
models = {
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\nTraining and evaluating models...")
best_model = None
best_score = float('inf')  # Lower MAE is better

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save the model
    joblib.dump(model, f'models/{name}_model.pkl')
    print(f"Model saved to models/{name}_model.pkl")
    
    # Track the best model based on MAE
    if mae < best_score:
        best_score = mae
        best_model = name

print(f"\nBest performing model: {best_model} with MAE: {best_score:.2f}")

# Load college and course data for allocation
print("\nLoading college and course data...")
college_data = pd.read_csv('processed_data/college_summary.csv')
course_data = pd.read_csv('processed_data/course_summary.csv')
ranking_data = pd.read_csv('processed_data/ranking_data_all.csv')

# Create a college allocation function
def allocate_college(predicted_rank):
    # Find colleges and courses where the predicted rank falls within the min and max rank range
    eligible_colleges = college_data[
        (college_data['min_rank'] <= predicted_rank) & 
        (college_data['max_rank'] >= predicted_rank)
    ].sort_values('min_rank')
    
    return eligible_colleges

# Create a course allocation function
def allocate_course(predicted_rank):
    # Find courses where the predicted rank falls within the min and max rank range
    eligible_courses = course_data[
        (course_data['min_rank'] <= predicted_rank) & 
        (course_data['max_rank'] >= predicted_rank)
    ].sort_values('min_rank')
    
    return eligible_courses

# Save the allocation functions
joblib.dump(allocate_college, 'models/college_allocation_function.pkl')
joblib.dump(allocate_course, 'models/course_allocation_function.pkl')

print("\nAllocation functions saved to models directory")

# Example prediction and allocation
print("\nExample prediction and allocation:")
# Use the best model for prediction
best_model_loaded = joblib.load(f'models/{best_model}_model.pkl')

# Sample input (physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet, 
#               physics_theory, chemistry_theory, mathematics_theory, biology_theory)
sample_input = np.array([[55, 58, 56, 57, 95, 90, 92, 88]])
predicted_rank = int(best_model_loaded.predict(sample_input)[0])
print(f"Predicted Rank: {predicted_rank}")

# Get eligible colleges and courses
eligible_colleges = allocate_college(predicted_rank)
eligible_courses = allocate_course(predicted_rank)

print("\nTop 5 Eligible Colleges:")
if not eligible_colleges.empty:
    print(eligible_colleges[['college_code', 'college_name', 'min_rank']].head())
else:
    print("No eligible colleges found for the predicted rank.")

print("\nTop 5 Eligible Courses:")
if not eligible_courses.empty:
    print(eligible_courses[['course_code', 'course_name', 'min_rank']].head())
else:
    print("No eligible courses found for the predicted rank.")

print("\nModel training and allocation system completed successfully!")