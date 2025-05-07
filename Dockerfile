FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure the models directory exists
RUN mkdir -p models

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]