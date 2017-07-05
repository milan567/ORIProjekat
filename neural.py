
__author__ = 'Nikolina'

import pandas
import numpy
from sklearn import model_selection
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

if __name__ == '__main__':

    url2 = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/ratingMovie.csv"
    names2 = ['userId', 'ourId', 'rating', 'timestamp', 'movieId', 'year', 'imdbRating', 'duration', 'director', 'actor1',
              'actor2', 'actor3', 'gross', 'country', 'budget', 'Mystery', 'Romance', 'Sci-Fi', 'Fantasy', 'Horror',
              'Film-Noir', 'Crime', 'Drama', 'Children', 'Animation', 'War', 'Adventure', 'Action', 'Comedy',
              'Documentary', 'Musical', 'Thriller', 'Western']
    ratings_loaded = pandas.read_csv(url2, names=names2)

    train_data, test_data = model_selection.train_test_split(ratings_loaded, test_size=0.20)

    array = train_data.values
    index_list = []
    for i in range(0,33):
        index_list.append(True)
    index_list[2] = False
    index_list[1] = False
    index_list[4] = False

    array[numpy.isnan(array) == True] = 0
    array[numpy.isfinite(array) == False] = 0
    X = array[:, index_list]
    y = array[:, [2]].flatten()
    y2 = map(str, y)
    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X)
    print("poceo")

    start = time.time()

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 30), random_state=1)
    clf.fit(X_train, y2)
    print("zavrsio")
    end = time.time()
    print(end - start)

    test_array = test_data.values
    test_array[numpy.isnan(test_array) == True] = 0
    test_array[numpy.isfinite(test_array) == False] = 0
    X_t = array[:,index_list]
    y_t = array[:, [2]].flatten()
    y_test = map(str, y_t)
    scaler.fit(X_t)
    X_test = scaler.transform(X_t)

    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    print 'MLP classifier RMSE: ' + str(sqrt(mean_squared_error(map(float,predictions), y_t)))