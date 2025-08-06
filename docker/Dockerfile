FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p models datasets

# Expose ports for the API and Dashboard
EXPOSE 8000
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app

# Create a script to run both services
RUN echo '#!/bin/bash\n\
python run_api.py &\n\
python run_dashboard.py' > /app/run.sh

RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
