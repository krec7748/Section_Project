import psycopg2
import sqlite3
import pandas as pd

host = "lallah.db.elephantsql.com"
database = "psqacqbs"
user = "psqacqbs"
password = "OTl5DwMRs8XKs3AOATNttNVzHoRMAIKd"

connection = psycopg2.connect(
    host=host,
    user=user,
    password=password,
    database=database
)

cur = connection.cursor()

cur.execute("DROP TABLE emotions;")

cur.execute("""CREATE TABLE emotions (
    ID INT,
    tweet_id INT,
    sentiment VARCHAR (12),
    content VARCHAR (256),
    sadness FLOAT,
    joy FLOAT,
    fear FLOAT,
    disgust FLOAT,
    anger FLOAT);
""")

connection.commit()

emotions= pd.read_csv("/Users/doukkim/Section_03/Section_Project/Section_03/emotions.csv", index_col = [0])
df_emotions = pd.DataFrame(emotions)
df_emotions_index = df_emotions.reset_index(drop = False)
df_emotions_index.to_csv("/Users/doukkim/Section_03/Section_Project/Section_03/emotions_index.csv", header = False, index = False)

query = """COPY emotions FROM STDIN DELIMITER','CSV;"""
with open("/Users/doukkim/Section_03/Section_Project/Section_03/emotions_index.csv") as f:
    cur.copy_expert(query, f)

connection.commit()
connection.close()