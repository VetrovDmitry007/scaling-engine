import pprint
import pymongo
import datetime
from pymongo import MongoClient

client = MongoClient(host='192.168.56.101', port=27010)

# Подключение к базе данных
db = client['test_database']

# Получаем коллекцию 'series' и сохраняем ее в переменной
series_collection = db['series']
print(series_collection)


post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}

# Отправка данных в базу
posts = db.pos
post_id = posts.insert_one(post).inserted_id

# Выводим список баз дынных
database_names = client.list_database_names()
print(database_names)

# Выводим список названий объектов в базе
ls_coll = db.list_collection_names()
print(ls_coll)

# Выводим первый объект из базы данных
dok = posts.find_one()
print(dok)

# Количество документов
print(posts.count_documents({}))

# Количество документов с конкретным элементом
print(posts.count_documents({"author": "Mike"}))

# Список элементов в коллекции
for post in posts.find():
  pprint.pprint(post)

# Удаление коллекции
series_collection.drop()

# Удаление бд
db.drop_database('test_database')

# Вывод по конкретному ключу
dok = posts.find_one({"author": "Mike"})
print(dok)
