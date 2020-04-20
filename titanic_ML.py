import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def Model(x_train, y_train, x_test, resultdata):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    score = accuracy_score(resultdata, prediction)
    return score


def Preprocessing(traindata, testdata, resultdata):
    resultdata = resultdata.drop(
        columns=['PassengerId'])
    traindata = traindata.drop(
        columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    testdata = testdata.drop(
        columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    traindata['Age'].fillna(int(traindata['Age'].mean()), inplace=True)
    testdata['Age'].fillna(int(testdata['Age'].mean()), inplace=True)
    traindata['Fare'].fillna(int(traindata['Fare'].mean()), inplace=True)
    testdata['Fare'].fillna(int(testdata['Fare'].mean()), inplace=True)
    traindata['Embarked'].fillna('S', inplace=True)
    testdata['Embarked'].fillna('S', inplace=True)
    # filled missing values in embarked to southampton because its probability is higher than other embarked locations

    wholedata = [testdata, traindata]
    for rows in wholedata:
        rows['Sex'] = rows['Sex'].map({'male': 0, 'female': 1})
        rows['Embarked'] = rows['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    testdata = testdata.values
    traindata = traindata.values
    traindata = preprocessing.scale(traindata)
    testdata = preprocessing.scale(testdata)

    y_train = traindata[0:, 0]  # Survived column
    for i in range(0, len(y_train)):
        y_train[i] = int(y_train[i])
    x_train = traindata[0:, 1:]  # Features
    x_test = testdata
    score = Model(x_train, y_train, x_test, resultdata)
    return score


if __name__ == "__main__":
    traindata = pd.read_csv("train.csv")
    testdata = pd.read_csv("test.csv")
    resultdata = pd.read_csv("gender_submission.csv")
    # contains solution, used to calculate accuracy
    score = Preprocessing(traindata, testdata, resultdata)
    print(' Accuracy: {0:0.1%}'.format(score))
