import csv
import os
import pandas as pd
from flask import Blueprint, render_template, request
from get_emotion import get_emotion
import pickle
import psycopg2

CSV_FILEPATH = os.path.join(os.getcwd(), 'emotions_result.csv') 

def import_data():
  with open(CSV_FILEPATH, newline='') as csvfile:
    dataset = csv.DictReader(csvfile)
    user_list = []
    for row in dataset:
      id_user = []
      id_user.append(row["sentiment"])
      id_user.append(row["content"])
      id_user.append(row["sadness"])
      id_user.append(row["joy"])
      id_user.append(row["fear"])
      id_user.append(row["disgust"])
      id_user.append(row["anger"])

      user_list.append(id_user)

    df = pd.DataFrame(user_list, columns = ["sentiment", "content", "sadness", "joy", "fear", "disgust", "anger"])

  return user_list, df

def return_predict_emotion_positivity(text):
    sadness_list, joy_list, fear_list, disgust_list, anger_list = get_emotion(data = [f"{text}"])
    data_emotion = {
            "sadness": sadness_list,
            "joy": joy_list,
            "fear": fear_list,
            "disgust": disgust_list,
            "anger": anger_list}
    
    df_for_predict = pd.DataFrame(data_emotion)

    model = pickle.load(open("RandomForest.pkl", "rb"))
    predict = model.predict(df_for_predict)

    df_text_info = pd.DataFrame({"sentiment": predict,
                                 "content": text})

    df_new = pd.concat([df_text_info, df_for_predict], axis = 1)

    #결과데이터 삽입(local)
    df_ori = pd.read_csv(CSV_FILEPATH, index_col = [0])
    df_ori_append = df_ori.append(df_new, ignore_index = True)
    df_ori_append.to_csv(CSV_FILEPATH)

    #결과데이터 삽입(DB)
    host = "lallah.db.elephantsql.com"
    database = "psqacqbs"
    user = "psqacqbs"
    password = "OTl5DwMRs8XKs3AOATNttNVzHoRMAIKd"

    connection = psycopg2.connect(
      host=host,
      user=user,
      password=password,
      database=database)

    cur = connection.cursor()
    
    id = len(df_ori)
    sentiment = df_new["sentiment"][0]
    content = df_new["content"][0]
    sadness = df_new["sadness"][0]
    joy = df_new["joy"][0]
    fear = df_new["fear"][0]
    disgust = df_new["disgust"][0]
    anger = df_new["anger"][0]

    cur.execute("INSERT INTO emotions (id, sentiment, content, sadness, joy, fear, disgust, anger) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (id, sentiment, content, sadness, joy, fear, disgust, anger))

    connection.commit()
    connection.close()

    #결과 데이터 json 형태로 return
    json_data = df_new.to_json(orient = 'records')

    return json_data
    
#Blueprint
main_bp = Blueprint('main', __name__)
predict_bp = Blueprint("predict", __name__)
db_bp = Blueprint("db", __name__)
db_sentiment_bp = Blueprint("db_sentiment", __name__ )

@main_bp.route('/')
def index():
  return render_template("index.html")

@predict_bp.route("/predict", methods=["GET"])
def get_text():
  from predict_emotion_positivity_app.views.main_views import import_data
  from predict_emotion_positivity_app.views.main_views import return_predict_emotion_positivity
  from urllib.parse import unquote

  df_emotion = import_data()[1]
  text = request.args.get("text", None)
  if text != None:
    text = unquote(text)
  
  if "+" in text:
    text = text.replace("+", " ")

  text_data_set = df_emotion["content"].tolist()

  if text is None:
    return "No text given", 400

  elif text in text_data_set:
    query = f'content == "{text}"'
    json_data = df_emotion.query(query).to_json(orient = 'records')
    return json_data, 200

  else:
    return return_predict_emotion_positivity(text), 200


@db_bp.route("/db")
def get_db():
  import json
  try:
    host = "lallah.db.elephantsql.com"
    database = "psqacqbs"
    user = "psqacqbs"
    password = "OTl5DwMRs8XKs3AOATNttNVzHoRMAIKd"

    connection = psycopg2.connect(
      host=host,
      user=user,
      password=password,
      database=database)

    cur = connection.cursor()

    cur.execute("""select json_build_object('id', id,
                                                   'sentiment', sentiment,
                                                   'content', content,
                                                   'sadness', sadness,
                                                   'joy', joy,
                                                   'fear', fear,
                                                   'disgust', disgust,
                                                   'anger', anger) from emotions """)

    json_data = cur.fetchall()

  finally:
    connection.close()

    return json.dumps(json_data)

@db_sentiment_bp.route("/db_sentiment")
def get_db_sentiment():
  from predict_emotion_positivity_app.views.main_views import get_db
  import json
  from urllib.parse import unquote
  sentiment = request.args.get("sentiment", None)
  json_data = json.loads(get_db())

  db_list = []
  for data in json_data:
    sentiment = unquote(sentiment)
    if data[0]["sentiment"] == sentiment:
      db_list.append(data[0])
  
  return json.dumps(db_list)