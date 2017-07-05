__author__ = 'Nikolina'

import pandas
import numpy
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression


if __name__=='__main__':
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
    y = array[:, [2]]

    lr = LinearRegression()
    lr.fit(X,y)

    test_array = test_data.values
    test_array[numpy.isnan(test_array) == True] = 0
    test_array[numpy.isfinite(test_array) == False] = 0
    X_test = array[:,index_list]
    y_test = array[:, [2]]
    predicted = lr.predict(X_test)
    print 'Regression RMSE: ' + str(sqrt(mean_squared_error(predicted,y_test)))