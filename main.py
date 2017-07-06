__author__ = 'Nikolina'

import pandas
import numpy
from sklearn import model_selection
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from math import sqrt
from sklearn.linear_model import LinearRegression
import os


def eval_cf(prediction, truth):
    prediction = prediction[truth.nonzero()].flatten()
    truth = truth[truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,truth))


def predict(ratings, similarity):
    rated_num = numpy.count_nonzero(ratings, axis=1)
    rated_num[rated_num == 0] = 6
    sum_user_rating = ratings.sum(axis=1)
    mean = sum_user_rating[:, numpy.newaxis]/rated_num[:, numpy.newaxis]
    ratings_diff = (ratings-mean)
    new_sim = similarity.copy()
    new_sim[new_sim > 0.7] = 0.000001
    predicted = mean + new_sim.dot(ratings_diff) / numpy.array([numpy.abs(new_sim).sum(axis=1)]).T
    return predicted



if __name__=='__main__':

    dir = os.path.dirname(__file__)
    filename1 = os.path.join(dir, 'ml-latest-small', 'ourMovies.csv')
    filename2 = os.path.join(dir, 'ml-latest-small', 'ratingMovie.csv')

    names_movie = ['ourId', 'movieId','year', 'rating', 'duration','director','actor1','actor2','actor3', 'gross', 'country', 'budget',
             'Mystery', 'Romance',
             'Sci-Fi','Fantasy', 'Horror', 'Film-Noir','Crime', 'Drama', 'Children', 'Animation', 'War', 'Adventure',
             'Action', 'Comedy', 'Documentary', 'Musical', 'Thriller', 'Western']
    movies = pandas.read_csv(filename1, names=names_movie)

    names2 = ['userId', 'ourId', 'rating', 'timestamp', 'movieId', 'year', 'imdbRating', 'duration', 'director', 'actor1',
              'actor2', 'actor3', 'gross', 'country', 'budget', 'Mystery', 'Romance', 'Sci-Fi', 'Fantasy', 'Horror',
              'Film-Noir', 'Crime', 'Drama', 'Children', 'Animation', 'War', 'Adventure', 'Action', 'Comedy',
              'Documentary', 'Musical', 'Thriller', 'Western']
    ratings_loaded = pandas.read_csv(filename2, names=names2)

    n_users = ratings_loaded.userId.unique().shape[0]
    train_data, test_data = model_selection.train_test_split(ratings_loaded, test_size=0.20)
    movie_num = movies.shape[0]

    train_data_matrix = numpy.zeros((n_users, movie_num))
    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]] = line[3]

    test_data_matrix = numpy.zeros((n_users, movie_num))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]] = line[3]

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    #predikcija za sve korisnike preko matrica
    predicted = predict(train_data_matrix, user_similarity)

    print 'User-based CF RMSE: ' + str(eval_cf(predicted, test_data_matrix))

    array = train_data.values
    index_list = []
    for i in range(0,33):
        index_list.append(True)
    index_list[2] = False
    index_list[3] = False
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

    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(30, 30), random_state=1)
    clf.fit(X_train, y2)
    #samo neuronska
    test_array_n = test_data.values
    test_array_n[numpy.isnan(test_array_n) == True] = 0
    test_array_n[numpy.isfinite(test_array_n) == False] = 0
    X_t_n = test_array_n[:,index_list]
    y_t_n = test_array_n[:, [2]].flatten()
    y_test_n = map(str, y_t_n)
    X_test_n = scaler.transform(X_t_n)
    predictions_n = clf.predict(X_test_n)
    print 'MLP classifier RMSE: ' + str(sqrt(mean_squared_error(map(float,predictions_n), y_t_n)))
    #print(classification_report(y_test_n,predictions_n))

    #samo regresija
    lr = LinearRegression()
    lr.fit(X,array[:,[2]])
    predictions_lr = lr.predict(X_t_n)
    print 'Regression RMSE: ' + str(sqrt(mean_squared_error(predictions_lr,test_array_n[:,[2]])))

    for i in range(0, len(predicted)):
        indexes = numpy.argpartition(predicted[i], -30)[-30:]
        frame = test_data[test_data.userId == i+1]
        for line in frame.itertuples():
            if line[2] not in indexes:
                ppt = test_data[(test_data.userId == i+1) & (test_data.ourId == line[2])].index.tolist()
                try:
                    test_data = test_data.drop(test_data.index[ppt[0]])
                except:
                    pass

    test_array_comb = test_data.values
    test_array_comb[numpy.isnan(test_array_comb) == True] = 0
    test_array_comb[numpy.isfinite(test_array_comb) == False] = 0
    X_t_comb = test_array_comb[:, index_list]
    y_t_comb = test_array_comb[:, [2]].flatten()
    y_test_comb = map(str, y_t_comb)
    scaler.fit(X_t_comb)
    X_test_comb = scaler.transform(X_t_comb)

    predictions_n_comb = clf.predict(X_test_comb)
    print 'CF + MLP classifier RMSE: ' + str(sqrt(mean_squared_error(map(float,predictions_n_comb), y_t_comb)))
    #print(classification_report(y_test_comb,predictions_n_comb))

    predictions_lr_comb = lr.predict(X_t_comb)
    print 'CF + Regression RMSE: ' + str(sqrt(mean_squared_error(predictions_lr_comb,test_array_comb[:,[2]])))