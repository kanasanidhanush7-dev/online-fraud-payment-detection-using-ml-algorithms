from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import base64
import io
import os
import joblib
import matplotlib.pyplot as plt
from .models import Register
from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import matplotlib
matplotlib.use('Agg')


global username
global X_train, X_test, y_train, y_test, X, Y, train_size
labels = ['Non-Fraud', 'Fraud']
accuracy = []
precision = []
recall = []
fscore = []

# Function to calculate all metrics


def calculateMetrics(algorithm, y_test, predict):
    a = (accuracy_score(y_test, predict) * 100)
    p = (precision_score(y_test, predict, average='macro') * 100)
    r = (recall_score(y_test, predict, average='macro') * 100)
    f = (f1_score(y_test, predict, average='macro') * 100)
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm


# Load saved ML model and preprocessing artifacts
dataset = pd.read_csv("Dataset/PS_20174392719_1491204439457_log.csv")

Y = dataset['isFraud'].to_numpy()
unique, count = np.unique(Y, return_counts=True)

dataset.drop(
    ['step', 'type', 'isFraud', 'isFlaggedFraud'],
    axis=1,
    inplace=True
)

# Load saved ML artifacts
label_encoders = joblib.load("model/label_encoders.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
rf = joblib.load("model/random_forest.pkl")

# Load original train/test data
data = np.load(
    "model/data.npy",
    allow_pickle=True
)

X_train, X_test, y_train, y_test = data
train_size = X_train.shape[0]

# Generate metrics using the saved model
predict = rf.predict(X_test)

calculateMetrics("RF", y_test, predict)

conf_matrix = confusion_matrix(y_test, predict)


def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})


def PredictAction(request):
    if request.method == 'POST':
        global rf, scaler, labels, dataset
        myfile = request.FILES['t1'].read()
        filename = request.FILES['t1'].name
        if os.path.exists('FraudApp/static/' + filename):
            os.remove('FraudApp/static/' + filename)
        with open('FraudApp/static/' + filename, "wb") as file:
            file.write(myfile)
        file.close()

        testData = pd.read_csv('FraudApp/static/' + filename)
        data = testData.values

        # FIXED: Dropping evaluating features ('isFraud', 'isFlaggedFraud') so exactly 7 columns match the training scaler
        testData.drop(['step', 'type', 'isFraud', 'isFlaggedFraud'],
                      axis=1, inplace=True, errors='ignore')

        for i in range(len(label_encoder)):
            le = label_encoder[i]
            # encode all str columns to numeric
            testData[le[0]] = pd.Series(
                le[1].transform(testData[le[0]].astype(str)))
        testData.fillna(dataset.mean(numeric_only=True), inplace=True)
        testData = scaler.transform(testData)
        predict = rf.predict(testData)

        output = '<table border=1 align=center width=100%><tr><th><font size="3" color="black">Test Data</th><th><font size="3" color="black">Detection Status</th></tr>'
        for i in range(len(predict)):
            if predict[i] == 0:
                output += '<tr><td><font size="3" color="black">' + \
                    str(data[i]) + \
                    '</td><td><font size="4" color="green">Normal Transaction</td></tr>'
            else:
                output += '<tr><td><font size="3" color="black">' + \
                    str(data[i]) + \
                    '</td><td><font size="4" color="red">Fraud Transaction</td></tr>'
        output += "</table></br></br></br></br>"
        context = {'data': output}
        return render(request, 'UserScreen.html', context)


def TrainML(request):
    algorithms = ['Random Forest']

    output = '<table border=1 align=center width=100%>'
    output += '<tr><th><font size="3" color="black">Algorithm Name</th>'
    output += '<th><font size="3" color="black">Accuracy</th>'
    output += '<th><font size="3" color="black">Precision</th>'
    output += '<th><font size="3" color="black">Recall</th>'
    output += '<th><font size="3" color="black">FSCORE</th></tr>'

    for i in range(len(algorithms)):
        output += '<tr><td><font size="3" color="black">' + str(algorithms[i])
        output += '</td><td><font size="3" color="black">' + str(accuracy[i])
        output += '</td><td><font size="3" color="black">' + str(precision[i])
        output += '</td><td><font size="3" color="black">' + str(recall[i])
        output += '</td><td><font size="3" color="black">' + str(fscore[i])
        output += '</td></tr>'

    output += '</table><br/><br/>'

    # Accuracy graph
    plt.figure(figsize=(6, 4))
    plt.bar(algorithms, accuracy)
    plt.title('Random Forest Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=15)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return render(
        request,
        'TrainML.html',
        {
            'data': output,
            'graph': graph
        }
    )


def LoadDataset(request):
    if request.method == 'GET':
        global unique, count, labels, dataset
        output = '<font size="3" color="black">Online Fraud Payment Detection Dataset Loaded</font><br/>'
        output += '<font size="3" color="blue">Total records found in Dataset = ' + \
            str(dataset.shape[0]) + '</font><br/>'
        output += '<font size="3" color="blue">Different Class Labels found in Dataset = ' + \
            str(labels) + '</font><br/><br/>'
        # visualizing class labels count found in dataset
        height = count
        bars = labels
        y_pos = np.arange(len(bars))
        plt.figure(figsize=(4, 3))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Imbalanced Dataset Class Label Graph")
        plt.ylabel("Count")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context = {'data': output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)


def BalancedData(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y, train_size
        output = '<font size="3" color="black">Smote Balancing Dataset Details</font><br/>'
        output += '<font size="3" color="blue">Training Size Before Applying SMOTE = ' + \
            str(train_size) + '</font><br/>'
        output += '<font size="3" color="blue">Training Size after Applying SMOTE = ' + \
            str(X_train.shape[0]) + '</font><br/><br/>'
        unique, count = np.unique(y_train, return_counts=True)
        height = count
        bars = labels
        y_pos = np.arange(len(bars))
        plt.figure(figsize=(4, 3))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Balanced Dataset After Applying SMOTE Graph")
        plt.ylabel("Count")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context = {'data': output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)


def UserLoginAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1').strip()
        password = request.POST.get('t2').strip()

        user = Register.objects.filter(username=username).first()

        if user:
            if user.password == password:
                return render(request, "UserScreen.html", {'data': 'Welcome ' + username})

        return render(request, 'UserLogin.html', {'data': 'Invalid username/password'})


def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1').strip()
        password = request.POST.get('t2').strip()
        contact = request.POST.get('t3')
        email = request.POST.get('t4')
        address = request.POST.get('t5')

        if Register.objects.filter(username=username).exists():
            output = "Username already exists"
        else:
            Register.objects.create(
                username=username,
                password=password,
                contact=contact,
                email=email,
                address=address
            )
            output = "Signup successful. Please login."

        return render(request, 'Register.html', {'data': output})


def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})


def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})


def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})


def RegisterPage(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})
