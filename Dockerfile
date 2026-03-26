# Use Python 3.10
FROM python:3.10

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file first to leverage Docker caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application files
COPY . .

# Create a cache directory for Transformers
RUN mkdir -p /code/cache && chmod -R 777 /code/cache
ENV TRANSFORMERS_CACHE=/code/cache

# Hugging Face Spaces specific environment variable
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
