FROM python:3.7.3-stretch

# Working Directory
WORKDIR /final

# Copy source code to working directory
COPY . application.py /final/

# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Run app.py at container launch
CMD ["python", "application.py"]