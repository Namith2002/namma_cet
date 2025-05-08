from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import os
import uvicorn
from contextlib import asynccontextmanager

app = FastAPI(
    title="KCET Rank Prediction API",
    description="API for predicting KCET ranks and allocating colleges based on exam scores",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and data at startup
model = None
scaler = None
college_data = None
course_data = None
cutoff_data = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and data at startup
    global model, scaler, college_data, course_data, cutoff_data
    
    # Load model and scaler
    try:
        model_paths = [
            'models/gradient_boosting_model.pkl',
            'models/random_forest_model.pkl',
            'models/linear_regression_model.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                print(f"Successfully loaded model from {path}")
                break
        
        if model is None:
            print("Warning: No valid model found. Predictions will not work.")
        
        # Load scaler if available
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Successfully loaded scaler")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    yield
    
    # Cleanup code here if needed

app = FastAPI(
    title="KCET Rank Prediction API",
    description="API for predicting KCET ranks and allocating colleges based on exam scores",
    version="1.0.0",
    lifespan=lifespan
)

# Load college and course data
try:
    college_data = pd.read_csv('processed_data/college_summary.csv')
    course_data = pd.read_csv('processed_data/course_summary.csv')
    print("Successfully loaded college and course data")
except Exception as e:
    print(f"Error loading college/course data: {e}")

# Load cutoff data
try:
    cutoff_paths = [
        'processed_data/college_cutoffs.csv',
        'processed_data/ranking_data_all.csv',
        'processed_data/college_summary.csv'
    ]
    
    for path in cutoff_paths:
        if os.path.exists(path):
            cutoff_data = pd.read_csv(path)
            print(f"Successfully loaded cutoff data from {path}")
            
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
    
    if cutoff_data is None and college_data is not None:
        print("No cutoff data found. Using college data as fallback.")
        cutoff_data = college_data.copy()
        cutoff_data['cutoff_rank'] = cutoff_data['min_rank']
        cutoff_data['category_code'] = 'GM'
        cutoff_data['category_type'] = 'General'
except Exception as e:
    print(f"Error loading cutoff data: {e}")

# Input models
class ScoreInput(BaseModel):
    physics_kcet: int = Field(..., ge=0, le=60, description="Physics KCET score (0-60)")
    chemistry_kcet: int = Field(..., ge=0, le=60, description="Chemistry KCET score (0-60)")
    mathematics_kcet: int = Field(..., ge=0, le=60, description="Mathematics KCET score (0-60)")
    biology_kcet: int = Field(..., ge=0, le=60, description="Biology KCET score (0-60)")
    physics_theory: int = Field(..., ge=0, le=100, description="Physics theory score (0-100)")
    chemistry_theory: int = Field(..., ge=0, le=100, description="Chemistry theory score (0-100)")
    mathematics_theory: int = Field(..., ge=0, le=100, description="Mathematics theory score (0-100)")
    biology_theory: int = Field(..., ge=0, le=100, description="Biology theory score (0-100)")
    category_code: str = Field("GM", description="Category code (GM, SCG, STG, 3AG)")
    category_type: str = Field("General", description="Category type (General, HK)")
    
    @validator('category_code')
    def validate_category(cls, v):
        valid_categories = ['GM', 'SCG', 'STG', '3AG']
        if v not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return v
    
    @validator('category_type')
    def validate_region(cls, v):
        valid_regions = ['General', 'HK']
        if v not in valid_regions:
            raise ValueError(f"Region must be one of {valid_regions}")
        return v

# Output models
class College(BaseModel):
    college_name: str
    college_code: str
    course_name: Optional[str] = None
    course_code: Optional[str] = None
    cutoff_rank: int

class PredictionResult(BaseModel):
    predicted_rank: int
    kcet_total: int
    theory_total: int
    combined_total: int
    kcet_percentage: float
    theory_percentage: float
    combined_percentage: float
    category_code: str
    category_type: str
    eligible_colleges: List[Dict[str, Any]]

# Helper functions
def predict_rank(input_data):
    """Predict rank using the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    # Prepare input data
    features = np.array([[
        input_data.physics_kcet, input_data.chemistry_kcet, 
        input_data.mathematics_kcet, input_data.biology_kcet,
        input_data.physics_theory, input_data.chemistry_theory, 
        input_data.mathematics_theory, input_data.biology_theory
    ]])
    
    # Apply scaling if scaler is available
    if scaler is not None:
        try:
            features = scaler.transform(features)
        except Exception as e:
            print(f"Warning: Could not apply scaling: {e}")
    
    # Predict rank
    try:
        predicted_rank = int(model.predict(features)[0])
        # Ensure rank is positive
        predicted_rank = max(1, predicted_rank)
        return predicted_rank
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting rank: {str(e)}")

def get_eligible_colleges(predicted_rank, category_code, category_type, limit=10):
    """Get eligible colleges based on predicted rank and category"""
    if cutoff_data is None:
        raise HTTPException(status_code=503, detail="College data not loaded. Please try again later.")
    
    try:
        # Try with exact category and type
        eligible_colleges = cutoff_data[
            (cutoff_data['category_code'] == category_code) &
            (cutoff_data['category_type'] == category_type) &
            (cutoff_data['cutoff_rank'] >= predicted_rank)
        ].sort_values('cutoff_rank')
        
        # If no colleges found, try with just the category
        if eligible_colleges.empty:
            eligible_colleges = cutoff_data[
                (cutoff_data['category_code'] == category_code) &
                (cutoff_data['cutoff_rank'] >= predicted_rank)
            ].sort_values('cutoff_rank')
        
        # If still no colleges found, try with General Merit category
        if eligible_colleges.empty and category_code != 'GM':
            eligible_colleges = cutoff_data[
                (cutoff_data['category_code'] == 'GM') &
                (cutoff_data['cutoff_rank'] >= predicted_rank)
            ].sort_values('cutoff_rank')
        
        # If still no colleges found, relax the rank constraint by 10%
        if eligible_colleges.empty:
            relaxed_rank = int(predicted_rank * 1.1)
            eligible_colleges = cutoff_data[
                cutoff_data['cutoff_rank'] >= relaxed_rank
            ].sort_values('cutoff_rank')
        
        # If still no colleges found, return top colleges regardless of rank
        if eligible_colleges.empty:
            eligible_colleges = cutoff_data.sort_values('cutoff_rank')
        
        # Return the top colleges
        return eligible_colleges.head(limit).to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting eligible colleges: {str(e)}")

# Add a new input model for college allocation
class AllocationInput(BaseModel):
    rank: int = Field(..., ge=1, le=250000, description="KCET Rank")
    category_code: str = Field("GM", description="Category code (GM, SCG, STG, 3AG)")
    category_type: str = Field("General", description="Category type (General, HK)")
    course_preference: Optional[str] = Field(None, description="Preferred course code")
    
    @validator('category_code')
    def validate_category(cls, v):
        valid_categories = ['GM', 'SCG', 'STG', '3AG']
        if v not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return v
    
    @validator('category_type')
    def validate_region(cls, v):
        valid_regions = ['General', 'HK']
        if v not in valid_regions:
            raise ValueError(f"Region must be one of {valid_regions}")
        return v

# Add a new output model for allocation results
class AllocationResult(BaseModel):
    rank: int
    category_code: str
    category_type: str
    course_preference: Optional[str]
    eligible_colleges: List[Dict[str, Any]]

# Add a new API endpoint for college allocation
@app.post("/allocate", response_model=AllocationResult)
async def allocate_colleges_api(allocation_input: AllocationInput, limit: int = Query(10, ge=1, le=50)):
    """
    Allocate colleges based on rank, category, region, and course preference
    
    - **allocation_input**: Rank and preference details
    - **limit**: Number of college recommendations to return (default: 10, max: 50)
    
    Returns eligible colleges based on the provided criteria
    """
    if cutoff_data is None:
        raise HTTPException(status_code=503, detail="College data not loaded. Please try again later.")
    
    try:
        # Start with filtering by rank and category/region
        query = (cutoff_data['cutoff_rank'] >= allocation_input.rank)
        
        if 'category_code' in cutoff_data.columns:
            query &= (cutoff_data['category_code'] == allocation_input.category_code)
        
        if 'category_type' in cutoff_data.columns:
            query &= (cutoff_data['category_type'] == allocation_input.category_type)
        
        # Filter by course preference if provided
        if allocation_input.course_preference and 'course_code' in cutoff_data.columns:
            query &= (cutoff_data['course_code'] == allocation_input.course_preference)
        
        eligible_colleges = cutoff_data[query].sort_values('cutoff_rank')
        
        # If no colleges found with exact criteria, try with relaxed constraints
        if eligible_colleges.empty:
            # Try without region constraint
            query = (cutoff_data['cutoff_rank'] >= allocation_input.rank)
            
            if 'category_code' in cutoff_data.columns:
                query &= (cutoff_data['category_code'] == allocation_input.category_code)
            
            if allocation_input.course_preference and 'course_code' in cutoff_data.columns:
                query &= (cutoff_data['course_code'] == allocation_input.course_preference)
            
            eligible_colleges = cutoff_data[query].sort_values('cutoff_rank')
        
        # If still no colleges found, try with general category
        if eligible_colleges.empty and allocation_input.category_code != 'GM':
            query = (cutoff_data['cutoff_rank'] >= allocation_input.rank)
            
            if 'category_code' in cutoff_data.columns:
                query &= (cutoff_data['category_code'] == 'GM')
            
            if allocation_input.course_preference and 'course_code' in cutoff_data.columns:
                query &= (cutoff_data['course_code'] == allocation_input.course_preference)
            
            eligible_colleges = cutoff_data[query].sort_values('cutoff_rank')
        
        # If still no colleges found, try without course preference
        if eligible_colleges.empty and allocation_input.course_preference:
            query = (cutoff_data['cutoff_rank'] >= allocation_input.rank)
            
            if 'category_code' in cutoff_data.columns:
                query &= (cutoff_data['category_code'] == allocation_input.category_code)
            
            if 'category_type' in cutoff_data.columns:
                query &= (cutoff_data['category_type'] == allocation_input.category_type)
            
            eligible_colleges = cutoff_data[query].sort_values('cutoff_rank')
        
        # If still no colleges found, relax the rank constraint by 10%
        if eligible_colleges.empty:
            relaxed_rank = int(allocation_input.rank * 1.1)
            query = (cutoff_data['cutoff_rank'] >= relaxed_rank)
            eligible_colleges = cutoff_data[query].sort_values('cutoff_rank')
        
        # Return the top colleges
        colleges = eligible_colleges.head(limit).to_dict('records')
        
        return AllocationResult(
            rank=allocation_input.rank,
            category_code=allocation_input.category_code,
            category_type=allocation_input.category_type,
            course_preference=allocation_input.course_preference,
            eligible_colleges=colleges
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error allocating colleges: {str(e)}")

# Add an endpoint to get available courses for dropdown/radio buttons
@app.get("/available-courses")
async def get_available_courses():
    """Get available courses for selection"""
    if course_data is None:
        raise HTTPException(status_code=503, detail="Course data not loaded")
    
    try:
        # Extract unique course codes and names
        courses = []
        if 'course_code' in course_data.columns and 'course_name' in course_data.columns:
            unique_courses = course_data[['course_code', 'course_name']].drop_duplicates()
            courses = unique_courses.to_dict('records')
        elif 'course_code' in cutoff_data.columns and 'course_name' in cutoff_data.columns:
            unique_courses = cutoff_data[['course_code', 'course_name']].drop_duplicates()
            courses = unique_courses.to_dict('records')
        
        return {"courses": courses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available courses: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to KCET Rank Prediction API"}

@app.post("/predict", response_model=PredictionResult)
async def predict_rank_api(score_input: ScoreInput, limit: int = Query(10, ge=1, le=50)):
    """
    Predict KCET rank and allocate colleges based on exam scores
    
    - **score_input**: KCET and theory exam scores
    - **limit**: Number of college recommendations to return (default: 10, max: 50)
    
    Returns predicted rank and eligible colleges
    """
    # Predict rank
    predicted_rank = predict_rank(score_input)
    
    # Calculate totals
    kcet_total = score_input.physics_kcet + score_input.chemistry_kcet + score_input.mathematics_kcet + score_input.biology_kcet
    theory_total = score_input.physics_theory + score_input.chemistry_theory + score_input.mathematics_theory + score_input.biology_theory
    combined_total = kcet_total + theory_total
    
    # Calculate percentages
    kcet_percentage = (kcet_total / 240) * 100
    theory_percentage = (theory_total / 400) * 100
    combined_percentage = (combined_total / 640) * 100
    
    # Get eligible colleges
    colleges = get_eligible_colleges(
        predicted_rank, 
        score_input.category_code, 
        score_input.category_type, 
        limit
    )
    
    # Return result
    return PredictionResult(
        predicted_rank=predicted_rank,
        kcet_total=kcet_total,
        theory_total=theory_total,
        combined_total=combined_total,
        kcet_percentage=round(kcet_percentage, 2),
        theory_percentage=round(theory_percentage, 2),
        combined_percentage=round(combined_percentage, 2),
        category_code=score_input.category_code,
        category_type=score_input.category_type,
        eligible_colleges=colleges
    )

@app.get("/categories")
async def get_categories():
    """Get available categories"""
    return {
        "categories": [
            {"code": "GM", "name": "General Merit"},
            {"code": "SCG", "name": "Scheduled Caste"},
            {"code": "STG", "name": "Scheduled Tribe"},
            {"code": "3AG", "name": "Other Backward Classes"}
        ],
        "regions": [
            {"code": "General", "name": "General"},
            {"code": "HK", "name": "Hyderabad-Karnataka"}
        ]
    }

@app.get("/colleges")
async def get_colleges(min_rank: int = 0, max_rank: int = 200000, limit: int = 100):
    """Get colleges within a rank range"""
    if college_data is None:
        raise HTTPException(status_code=503, detail="College data not loaded")
    
    filtered_colleges = college_data[
        (college_data['min_rank'] >= min_rank) & 
        (college_data['max_rank'] <= max_rank)
    ].sort_values('min_rank').head(limit)
    
    return {"colleges": filtered_colleges.to_dict('records')}

@app.get("/courses")
async def get_courses(min_rank: int = 0, max_rank: int = 200000, limit: int = 100):
    """Get courses within a rank range"""
    if course_data is None:
        raise HTTPException(status_code=503, detail="Course data not loaded")
    
    filtered_courses = course_data[
        (course_data['min_rank'] >= min_rank) & 
        (course_data['max_rank'] <= max_rank)
    ].sort_values('min_rank').head(limit)
    
    return {"courses": filtered_courses.to_dict('records')}

@app.get("/health")
async def health_check():
    """Check if the API is healthy"""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "college_data_loaded": college_data is not None,
        "course_data_loaded": course_data is not None,
        "cutoff_data_loaded": cutoff_data is not None
    }
    return status

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
