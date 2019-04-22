FROM python:3.7 as PythonBase

# Python Libraries
WORKDIR /music
COPY requirements.txt /music/
RUN pip install -r requirements.txt

# Data and Scripts
COPY data /music/data
COPY scripts /music/scripts

# Run pipeline
CMD ["bash", "scripts/run_all.sh"]
