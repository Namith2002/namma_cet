import pandas as pd
import numpy as np
import joblib
import os

# Load college and course data
try:
    college_data = pd.read_csv('processed_data/college_summary.csv')
    course_data = pd.read_csv('processed_data/course_summary.csv')
except Exception as e:
    print(f"Error loading data files: {e}")
    print("Please make sure the required CSV files exist in the processed_data directory.")

# Define allocation functions directly in this file
def get_eligible_colleges(predicted_rank, limit=10):
    """
    Find colleges where the predicted rank falls within the min and max rank range
    Returns top 10 colleges by default
    """
    eligible_colleges = college_data[
        (college_data['min_rank'] <= predicted_rank) & 
        (college_data['max_rank'] >= predicted_rank)
    ].sort_values('min_rank')
    
    return eligible_colleges.head(limit)

def get_eligible_courses(predicted_rank, limit=10):
    """
    Find courses where the predicted rank falls within the min and max rank range
    Returns top 10 courses by default
    """
    eligible_courses = course_data[
        (course_data['min_rank'] <= predicted_rank) & 
        (course_data['max_rank'] >= predicted_rank)
    ].sort_values('min_rank')
    
    return eligible_courses.head(limit)

def predict_rank_and_allocate(physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet,
                             physics_theory, chemistry_theory, mathematics_theory, biology_theory,
                             model_name='gradient_boosting', limit=10):
    """
    Predict the rank based on KCET and theory exam marks and allocate colleges and courses.
    
    Parameters:
    - physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet: KCET exam marks (out of 60)
    - physics_theory, chemistry_theory, mathematics_theory, biology_theory: Theory exam marks (out of 100)
    - model_name: The model to use for prediction ('linear_regression', 'random_forest', or 'gradient_boosting')
    - limit: Number of colleges and courses to return (default: 10)
    
    Returns:
    - predicted_rank: The predicted rank
    - eligible_colleges: DataFrame of eligible colleges (top 10 by default)
    - eligible_courses: DataFrame of eligible courses (top 10 by default)
    """
    # Check if models directory exists
    if not os.path.exists('models'):
        print("Error: Models directory not found. Please run train_model.py first.")
        return None, None, None
    
    # Load the model
    model_path = f'models/{model_name}_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model {model_name} not found. Please run train_model.py first.")
        return None, None, None
    
    model = joblib.load(model_path)
    
    # Prepare input data
    input_data = np.array([[physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet,
                           physics_theory, chemistry_theory, mathematics_theory, biology_theory]])
    
    # Predict rank
    predicted_rank = int(model.predict(input_data)[0])
    
    # Get eligible colleges and courses using the functions defined in this file
    eligible_colleges = get_eligible_colleges(predicted_rank, limit)
    eligible_courses = get_eligible_courses(predicted_rank, limit)
    
    return predicted_rank, eligible_colleges, eligible_courses

def load_model():
    """Load the trained model and scaler"""
    if not os.path.exists('models'):
        print("Error: Models directory not found. Please run train_model.py first.")
        return None, None
    
    # Try different model names if the default one doesn't exist
    model_paths = [
        'models/gradient_boosting_model.pkl',
        'models/random_forest_model.pkl',
        'models/linear_regression_model.pkl'
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"Successfully loaded model from {path}")
                break
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
    
    if model is None:
        print("Error: No valid model found. Please run train_model.py first.")
        return None, None
    
    # Assuming there's a scaler saved, if not, return None for scaler
    scaler_path = 'models/scaler.pkl'
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading scaler: {e}")
            print("Continuing without scaler...")
    
    return model, scaler

def load_cutoff_data():
    """Load the cutoff data for college allocation"""
    try:
        # Try different possible cutoff data files
        cutoff_paths = [
            'processed_data/college_cutoffs.csv',
            'processed_data/ranking_data_all.csv',
            'processed_data/college_summary.csv'
        ]
        
        cutoff_data = None
        for path in cutoff_paths:
            if os.path.exists(path):
                cutoff_data = pd.read_csv(path)
                print(f"Successfully loaded cutoff data from {path}")
                
                # Check if the required columns exist
                required_columns = ['college_code', 'college_name', 'course_code', 'course_name', 'cutoff_rank']
                
                # If using college_summary.csv, adapt the column names
                if path == 'processed_data/college_summary.csv':
                    # Map min_rank to cutoff_rank if needed
                    if 'min_rank' in cutoff_data.columns and 'cutoff_rank' not in cutoff_data.columns:
                        cutoff_data['cutoff_rank'] = cutoff_data['min_rank']
                
                # Add category columns if they don't exist
                if 'category_code' not in cutoff_data.columns:
                    cutoff_data['category_code'] = 'GM'
                if 'category_type' not in cutoff_data.columns:
                    cutoff_data['category_type'] = 'General'
                
                break
        
        if cutoff_data is None:
            print("Error: No cutoff data found. Using a simplified allocation method.")
            # Create a simplified version using college_data
            cutoff_data = college_data.copy()
            cutoff_data['cutoff_rank'] = cutoff_data['min_rank']
            cutoff_data['category_code'] = 'GM'
            cutoff_data['category_type'] = 'General'
            
            # Join with course data if available
            if 'course_data' in globals() and not course_data.empty:
                # Assuming there's a relationship between college and course
                # This is a simplified approach
                cutoff_data = cutoff_data.merge(
                    course_data[['course_code', 'course_name']],
                    left_on='college_code',  # Adjust these join columns based on your data
                    right_on='course_code',
                    how='left'
                )
        
        return cutoff_data
    except Exception as e:
        print(f"Error loading cutoff data: {e}")
        return None

def predict_rank(model, scaler, physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet,
                physics_theory, chemistry_theory, mathematics_theory, biology_theory):
    """Predict the rank using the loaded model and scaler"""
    # Prepare input data
    input_data = np.array([[physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet,
                           physics_theory, chemistry_theory, mathematics_theory, biology_theory]])
    
    # Apply scaling if scaler is provided
    if scaler is not None:
        try:
            input_data = scaler.transform(input_data)
        except Exception as e:
            print(f"Warning: Could not apply scaling: {e}")
            print("Continuing with unscaled data...")
    
    # Predict rank
    try:
        predicted_rank = int(model.predict(input_data)[0])
        # Ensure rank is positive
        predicted_rank = max(1, predicted_rank)
        return predicted_rank
    except Exception as e:
        print(f"Error predicting rank: {e}")
        # Return a default rank as fallback
        return 10000

def allocate_colleges(cutoff_data, predicted_rank, category_code='GM', category_type='General', limit=10):
    """
    Allocate multiple colleges based on predicted rank and category
    Returns at least 10 colleges if available
    """
    try:
        # Check if the required columns exist
        required_columns = ['cutoff_rank', 'college_name', 'course_name', 'college_code', 'course_code']
        missing_columns = [col for col in required_columns if col not in cutoff_data.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns in cutoff data: {missing_columns}")
            print("Using available columns for allocation...")
        
        # Filter cutoff data based on category and rank if those columns exist
        if 'category_code' in cutoff_data.columns and 'category_type' in cutoff_data.columns:
            eligible_colleges = cutoff_data[
                (cutoff_data['category_code'] == category_code) &
                (cutoff_data['category_type'] == category_type) &
                (cutoff_data['cutoff_rank'] >= predicted_rank)
            ].sort_values('cutoff_rank')
        else:
            # Simplified filtering if category columns don't exist
            eligible_colleges = cutoff_data[
                cutoff_data['cutoff_rank'] >= predicted_rank
            ].sort_values('cutoff_rank')
        
        # Return the top colleges (limited by the parameter)
        if not eligible_colleges.empty:
            # If we have fewer than the requested limit, return all available
            return eligible_colleges.head(limit).to_dict('records')
        return []
    except Exception as e:
        print(f"Error in college allocation: {e}")
        return []

def main():
    print("\n====================================================")
    print("       KCET RANK PREDICTION & COLLEGE ALLOCATION")
    print("====================================================")
    
    # Load model and data
    model, scaler = load_model()
    if model is None:
        return
    
    cutoff_data = load_cutoff_data()
    if cutoff_data is None:
        return
    
    try:
        # Get user input for scores
        print("\nPlease enter your KCET exam scores (out of 60):")
        physics_kcet = int(input("Physics: "))
        chemistry_kcet = int(input("Chemistry: "))
        mathematics_kcet = int(input("Mathematics: "))
        biology_kcet = int(input("Biology: "))
        
        print("\nPlease enter your Theory exam scores (out of 100):")
        physics_theory = int(input("Physics: "))
        chemistry_theory = int(input("Chemistry: "))
        mathematics_theory = int(input("Mathematics: "))
        biology_theory = int(input("Biology: "))
        
        # Validate input
        if any(score < 0 or score > 60 for score in [physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet]):
            print("Error: KCET scores must be between 0 and 60.")
            return
        
        if any(score < 0 or score > 100 for score in [physics_theory, chemistry_theory, mathematics_theory, biology_theory]):
            print("Error: Theory scores must be between 0 and 100.")
            return
        
        # Get category information
        print("\nPlease select your category:")
        print("1. General Merit (GM)")
        print("2. Scheduled Caste (SC)")
        print("3. Scheduled Tribe (ST)")
        print("4. Other Backward Classes (OBC)")
        category_choice = int(input("Enter choice (1-4): "))
        
        category_mapping = {
            1: 'GM',
            2: 'SCG',
            3: 'STG',
            4: '3AG'  # Using 3AG as a representative OBC category
        }
        
        if category_choice not in category_mapping:
            print("Invalid choice. Using General Merit (GM) as default.")
            category_code = 'GM'
        else:
            category_code = category_mapping[category_choice]
        
        # Get region information
        print("\nPlease select your region:")
        print("1. General")
        print("2. Hyderabad-Karnataka (HK)")
        region_choice = int(input("Enter choice (1-2): "))
        
        category_type = 'General' if region_choice == 1 else 'HK'
        
        # Predict rank
        predicted_rank = predict_rank(model, scaler, physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet,
                                    physics_theory, chemistry_theory, mathematics_theory, biology_theory)
        
        # Calculate totals
        kcet_total = physics_kcet + chemistry_kcet + mathematics_kcet + biology_kcet
        theory_total = physics_theory + chemistry_theory + mathematics_theory + biology_theory
        
        # Display results
        print("\n====================================================")
        print("                    RESULTS")
        print("====================================================")
        print(f"KCET Total: {kcet_total}/240 ({kcet_total/240*100:.2f}%)")
        print(f"Theory Total: {theory_total}/400 ({theory_total/400*100:.2f}%)")
        print(f"Combined Total: {kcet_total + theory_total}/640 ({(kcet_total + theory_total)/640*100:.2f}%)")
        print(f"\nPredicted Rank: {predicted_rank}")
        print(f"Category: {category_code}, Region: {category_type}")
        
        # In the main() function, replace the college allocation section with:
        # Get multiple college recommendations (at least 10)
        colleges = allocate_colleges(cutoff_data, predicted_rank, category_code, category_type, limit=10)
        
        if not colleges:
            print("\nNo college allocation found for your predicted rank and category.")
            print("You may want to consider:")
            print("1. Improving your scores")
            print("2. Exploring other categories")
            print("3. Checking for colleges in the General region")
            print("4. Applying through management quota if available")
        else:
            print("\n====================================================")
            print("               COLLEGE ALLOCATIONS")
            print("====================================================")
            print(f"Top {len(colleges)} recommended colleges for your rank:")
            print("\n")
            
            for i, college in enumerate(colleges, 1):
                print(f"Option {i}:")
                for key, value in college.items():
                    if key in ['college_name', 'course_name', 'college_code', 'course_code', 'cutoff_rank']:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
                print()
            
    except ValueError:
        print("Error: Please enter valid numeric values for all scores.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# Example usage
if __name__ == "__main__":
    main()