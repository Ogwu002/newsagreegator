FROM python:3

LABEL maintainer="Joseph"
#set the working directory in the container
WORKDIR /mlapp
#copy the current directory contents into the container at /app
COPY . .
COPY model/rental_prediction_model.pkl ./model.pkl
#install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "./app.py" ]

EXPOSE 5000
