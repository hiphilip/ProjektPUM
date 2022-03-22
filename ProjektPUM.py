import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, make_scorer, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GroupKFold, StratifiedKFold
import optuna

dane = pd.read_csv("Final database.csv")

bazaPL = dane.loc[dane["Country"] == "Poland"]
labels = bazaPL["Top50_dummy"]
bazaPL.shape

cechy = bazaPL.loc[:, "danceability":"valence"]
X_train, X_test, y_train, y_test = train_test_split(cechy, labels, test_size=0.2, random_state=42, stratify=labels)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# print(X_train)
X_train = X_train.reshape(np.shape(X_train)[0], -1)
X_test = X_test.reshape(np.shape(X_test)[0], -1)

# pętle zamieniające tę wartość na 0
# nie wiem skąd one występują, możliwe że przy zmianie na numpy.array
for i in range(len(X_train)):
    for k in range(10):
        if X_train[i][k] == 'n\x07':
            X_train[i][k] = 0
i = 0

k = 0
for i in range(len(y_train)):
    if y_train[i] == 'n\x07':
        y_train[i] = 0

scoring = {'f1_macro': make_scorer(f1_score, average='macro')}


def objective(trial, model, get_space, X, y):
    model_space = get_space(trial)

    mdl = model(**model_space)
    scores = cross_validate(mdl, X, y, scoring=scoring, cv=StratifiedKFold(n_splits=5), return_train_score=True)

    return np.mean(scores['test_f1_macro'])


model = KNeighborsClassifier


def get_space(trial):
    space = {"n_neighbors": trial.suggest_int('n_neighbors', 3, 100)}
    return space


trials = 30

study = optuna.create_study(direction='maximize')
study.optimize(lambda x: objective(x, model, get_space, X_train, y_train), n_trials=trials)

clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(preds)

pickle.dump(clf, open(b"knn_model", "wb"))
model = pickle.load(open("knn_model", "rb"))

preds = model.predict(X_test)

confusion_matrix(y_test, preds)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print('TN = ', tn, ' FP = ', fp, ' FN = ', fn, ' TP = ', tp)

accuracy = accuracy_score(y_test, preds)
recall = recall_score(y_test, preds)
precision = precision_score(y_test, preds)
f1 = f1_score(y_test, preds)
print('accuracy = ', accuracy, ' recall = ', recall, ' precision = ', precision, ' F1 = ', f1)

###################################
### LAS LOSOWY ####################
###################################

scoring = {'f1_macro': make_scorer(f1_score, average='macro')}

model = RandomForestClassifier(random_state=42)
scores = cross_validate(model, X_train, y_train, scoring=scoring)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average="macro"))

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('accuracy = ', accuracy, ' recall = ', recall, ' precision = ', precision, 'F1 = ', f1)

np.argmax(model.feature_importances_), np.argmin(model.feature_importances_)
