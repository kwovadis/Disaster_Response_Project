# Disaster_Response_Project
Analyzing messages data for disaster response via an ETL pipeline (Extract, Transform, and Load), which read, cleaned and stored the dataset in a SQLite database. 

A machine learning pipeline was then created using scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification).
Finally, the results are displayed using a Flask Web App. 

![image](https://user-images.githubusercontent.com/98783364/185207321-14803f5c-bd42-4b46-ab09-8640e486e682.png)

To run the pipelines and the Flask Web App, clone the repository and run the following commands: 
```
data folder  >>> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
model folder >>> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
app folder   >>> python run.py 
```

The repository contains the following files:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app
- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
- models
|- train_classifier.py
- README.md
```
