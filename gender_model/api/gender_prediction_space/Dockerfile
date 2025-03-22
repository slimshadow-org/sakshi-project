FROM python:3.9

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Create a non-root user to run the application
RUN useradd -m appuser

# Create directories with appropriate permissions
RUN mkdir -p /code/output && \
    chown -R appuser:appuser /code

COPY --chown=appuser:appuser ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# For headless matplotlib
ENV MPLBACKEND=Agg

COPY --chown=appuser:appuser . /code/

# Switch to the non-root user
USER appuser

# Make sure the app.py file is correctly named
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]