import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions

def get_emotion(data, sadness_list = [], joy_list = [], fear_list = [], disgust_list = [], anger_list = [], sentence_list = []):
    #IBM_Natural_Language_Understanding API >>> Extracting emotions from text
    #data = list or Series ex) data = ["I love an apple.", "I hate the smell of cigarettes."]
    
    API_KEY = "Mvol2ZlsQFok7RAf72KG5NFs4ehiEDjIhrqKx3BeiFAi"
    URL = "https://api.kr-seo.natural-language-understanding.watson.cloud.ibm.com/instances/f0484008-e8aa-4564-afa9-6e3d7f51639b"

    authenticator = IAMAuthenticator(API_KEY)
    natural_language_understanding = NaturalLanguageUnderstandingV1(version='2021-08-01',authenticator=authenticator)
    natural_language_understanding.set_service_url(URL)
    try:
        for sentence in data[len(anger_list):]:
            response = natural_language_understanding.analyze(text = sentence, features=Features(emotion=EmotionOptions()), language='en').get_result()
            json_emotion = json.loads(json.dumps(response))["emotion"]["document"]["emotion"]

            sentence_list.append(sentence)
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
        return sentence_list, sadness_list, joy_list, fear_list, disgust_list, anger_list