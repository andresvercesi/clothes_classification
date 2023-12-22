#Select python version image
FROM python:3.10.2
#Disable pip upgrade message
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

ENV PYTHONUNBUFFERED=1
#Set work folder
WORKDIR /app
#Copy the requirements file
COPY requirements.txt .
#Create and activate virtual environment
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"
#Install requirements
RUN pip install -r requirements.txt
#Copy files to work folder
COPY . . 
#Set port to expose app
EXPOSE 8000
#Command for run uvicorn
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]


