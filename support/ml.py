

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
dataset = pd.read_csv('clean_data.csv')
corpus = []

nltk.download('stopwords')

for i in range(0, 1000):
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text = [ps.stem(word) for word in text if word not in set(all_stopwords)]
    text = ' '.join(text)
    corpus.append(text)

    cv = CountVectorizer(max_features=15000)
    x = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:1000, 0].values

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)
    # if request.method == 'POST':
        # message = request.form['review']
        # data = [message]
    data= "vaccination center sucks"
    data = data.lower()
    data = [ps.stem(data) for word in text if word not in set(all_stopwords)]
    vect = cv.transform(data).toarray()
    my_prediction = classifier.predict(vect)
    if my_prediction !=1:
        print("this seems good")
    else:
        print( "this needs to be changed" )


