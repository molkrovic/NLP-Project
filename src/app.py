import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

url = 'https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv'

df_raw = pd.read_csv(url)

df = df_raw.copy()

df.drop_duplicates()

df['url'] = df['url'].str.lower()

def eliminar_https(texto):
    return re.sub(r'(https://www|https://)', '', texto)

def caracteres_no_alfanumericos(texto):
    return re.sub("(\\W)+"," ", texto)

def esp_multiple(texto):
    return re.sub(' +', ' ',texto)

df['url_limpia'] = df['url'].apply(eliminar_https).apply(caracteres_no_alfanumericos).apply(esp_multiple)

df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x==True else 0)

df.to_csv('../data/processed/df_processed.csv')
print('Se guardó el csv procesado.')
print()

X = df['url_limpia']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=12)

vec = CountVectorizer()

X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()

classifier_hp = SVC(C=10, gamma=0.1, random_state=1234)

classifier_hp.fit(X_train, y_train)

filename = '../models/modelo_NLP.sav'
pickle.dump(classifier_hp, open(filename, 'wb'))
print('Se guardó el modelo.')
print()

predictions_hp = classifier_hp.predict(X_test)
print('Métricas:')
print(classification_report(y_test, predictions_hp))