# Use the official image as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_dev.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the app
CMD ["streamlit", "run", "streamlit_app/app.py"]
