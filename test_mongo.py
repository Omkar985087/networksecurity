from pymongo.mongo_client import MongoClient
from urllib.parse import quote_plus

username = quote_plus("Omkar9850")
password = quote_plus("Omkar@9850")   # @ will be encoded automatically

uri = f"mongodb+srv://{username}:{password}@cluster0.vehgjny.mongodb.net/?appName=Cluster0"

client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
