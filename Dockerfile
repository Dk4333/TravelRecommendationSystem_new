# Use a specific, lightweight version of Python as the base image
FROM python:3.9-slim

# Set the working directory inside the container to /app
# This is where your code will live
WORKDIR /app

# Copy the requirements.txt file into the container's working directory
COPY requirements.txt .

# Install all the Python libraries listed in requirements.txt
# --no-cache-dir makes the final image smaller
# --upgrade pip ensures you have the latest version of pip
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Download the NLTK data (stopwords and wordnet) during the build process
# This prevents it from downloading every time the app starts
RUN python -m nltk.downloader stopwords wordnet

# Copy all the other files from your local project folder (app.py, .csv files, etc.)
# into the container's /app directory
COPY . .

# Tell the container that your application will listen on port 7860
# This is the default port Hugging Face Spaces expects
EXPOSE 7860

# The command to run when the container starts.
# This uses Flask's built-in command to run the web server.
# It's set to be accessible from outside the container on the exposed port.
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
