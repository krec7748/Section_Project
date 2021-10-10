import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions

def get_emotion(data):
    #IBM_Natural_Language_Understanding API >>> Extracting emotions from text
    #data = list or Series ex) data = ["I love an apple.", "I hate the smell of cigarettes."]

    sadness_list = []
    joy_list = []
    fear_list = []
    disgust_list = []
    anger_list = []
    
    API_KEY = "JTc1E1rgxjWb4pQv_DpaajRzCpWp7zLY-bAYR9nf4v3-"
    URL = "https://api.us-east.natural-language-understanding.watson.cloud.ibm.com/instances/b8eba1c1-126f-49f3-b1eb-9e187b9db2e7"

    authenticator = IAMAuthenticator(API_KEY)
    natural_language_understanding = NaturalLanguageUnderstandingV1(version='2021-08-01',authenticator=authenticator)
    natural_language_understanding.set_service_url(URL)
    try:
        for sentence in data[len(anger_list):]:
            response = natural_language_understanding.analyze(text = sentence, features=Features(emotion=EmotionOptions()), language='en').get_result()
            json_emotion = json.loads(json.dumps(response))["emotion"]["document"]["emotion"]

            sadness_list.append(json_emotion["sadness"])
            joy_list.append(json_emotion["joy"])
            fear_list.append(json_emotion["fear"])
            disgust_list.append(json_emotion["disgust"])
            anger_list.append(json_emotion["anger"])
    except:
        print(len(anger_list))
        if len(anger_list) == len(data):
            pass
        else:
            print("API error") #get_emotion() #유료버전일 때 사용.
    finally:
        return sadness_list, joy_list, fear_list, disgust_list, anger_list