import pandas as pd
from pandas.core.frame import DataFrame
import pickle

#데이터 불러오기
df_emotion = pd.read_csv("/Users/doukkim/Section_03/Section_Project/Section_03/emotions.csv", index_col = [0])

#기본 전처리
df_emotion.drop(["tweet_id", "content"], axis = 1, inplace = True)

def reduce_sentiment_type_4(string):
    #df_emotion의 target class를 축소시키기 위한 함수(13 >> 4)
    if string in ["enthusiasm", "happiness", "love", "surprise", "fun"]:
        return "positive"
    elif string in ["boredom", "worry", "relief", "sadness", "empty"]:
        return "negative"
    elif string in ["anger", "hate"]:
        return "very negative"
    else:
        return string

df_emotion["sentiment"] = df_emotion["sentiment"].apply(reduce_sentiment_type_4)

#분석
target = "sentiment"
features = df_emotion.drop(target, axis = 1).columns

##데이터 셋 나누기
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_emotion, test_size = 0.2, random_state = 1, stratify = df_emotion[target])
train, val = train_test_split(train, test_size = 0.2, random_state = 1, stratify = train[target])

X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]

##학습
#필요한 라이브러리 불러오기
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report

#표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#Model 생성
#RandomForestClassifier
forest = RandomForestClassifier(random_state = 1
                                , bootstrap = 500
                                , n_jobs = -1
                                , criterion = "entropy"
                                , max_depth = 10)

#OneVsRestClassifier_RandomForest
one_rest_forest = OneVsRestClassifier(forest, n_jobs=-1)

#모델 학습
forest.fit(X_train, y_train)
one_rest_forest.fit(X_train, y_train)

#예측값 도출
y_pred_forest_train = forest.predict(X_train)
y_pred_forest_val = forest.predict(X_val)

y_pred_onerest_forest_train = one_rest_forest.predict(X_train)
y_pred_onerest_forest_val = one_rest_forest.predict(X_val)


#평가
def evaluate_model ():
    #predict
    y_pred_forest_train = forest.predict(X_train)
    y_pred_forest_val = forest.predict(X_val)
    y_pred_forest_test = forest.predict(X_test)

    y_pred_onerest_forest_train = one_rest_forest.predict(X_train)
    y_pred_onerest_forest_val = one_rest_forest.predict(X_val)
    y_pred_onerest_forest_test = one_rest_forest.predict(X_test)

    #evaluate accuracy
    forest_train_accuracy = accuracy_score(y_train, y_pred_forest_train)
    forest_val_accuracy = accuracy_score(y_val, y_pred_forest_val)
    forest_test_accuracy = accuracy_score(y_test, y_pred_forest_test)
    forest_val_report = classification_report(y_val, y_pred_forest_val)

    onerest_forest_train_accuracy = accuracy_score(y_train, y_pred_onerest_forest_train)
    onerest_forest_val_accuracy = accuracy_score(y_val, y_pred_onerest_forest_val)
    one_rest_forest_test_accuracy = accuracy_score(y_test, y_pred_onerest_forest_test)
    onerest_forest_val_report = classification_report(y_val, y_pred_onerest_forest_val)

    #return
    if forest_val_accuracy >= onerest_forest_val_accuracy:
        print("RandomForest의 train, val, test의 accuracy와 classification report(validation data set)가 반환됩니다.")
        return forest_train_accuracy, forest_val_accuracy, forest_test_accuracy, forest_val_report
    else:
        print("OneVsRestRandomForest의 train, val, test의 accuracy와 classification report(validation data set)가 반환됩니다.")
        return onerest_forest_train_accuracy, onerest_forest_val_accuracy, one_rest_forest_test_accuracy, onerest_forest_val_report


#Confusion Matrix
#Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

# Compute confusion matrix
def single_model(model):
    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    conf_mx = confusion_matrix(y_pred, y_val, normalize = "true")
    return conf_mx

#plot confusion matrix
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, model_name):

    plt.figure(figsize=(10,15))
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix: '+ model_name, fontsize=10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = 'f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=10)

    plt.tight_layout()
    plt.ylabel('True label',fontsize=10,color='black')
    plt.xlabel('Predicted label',fontsize=10,color='black')
    np.set_printoptions(precision=2)
    
    return plt.show()

#Confusion Martrix 출력
#classes = ["negative", "neutral", "positive", "very negative"]
#plot_confusion_matrix(single_model(one_rest_forest), classes,'One_Vs_Rest_RandomForset Model')

from get_emotion import get_emotion
def return_predict_emotion_positivity(data):
    sentence_list, sadness_list, joy_list, fear_list, disgust_list, anger_list = get_emotion(data)
    data = {"sadness": sadness_list,
            "joy": joy_list,
            "fear": fear_list,
            "disgust": disgust_list,
            "anger": anger_list}
    
    df_X = pd.DataFrame(data)
    #df_X_scaled = scaler.transform(df_X)

    y_pred = one_rest_forest.predict(df_X)
    y_pred_list = y_pred.tolist()

    df_X.insert(0, "text", sentence_list)
    df_X["emotion_positivity"] = y_pred_list

    return df_X

pickle.dump(forest, open("RandomForest.pkl", "wb"))