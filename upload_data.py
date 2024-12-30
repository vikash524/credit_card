from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# uniform resource indentifier
uri = "mongodb+srv://vikashchauhan:vikashchauhan@cluster0.xdsknwb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# create database name and collection name
DATABASE_NAME="project2"
COLLECTION_NAME="creditcard"

# read the data as a dataframe
df=pd.read_csv(r"D:\credit card\notebooks\creditCardFraud_28011964_120214 (1).csv")


# Convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

#now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
