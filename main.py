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


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

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
    url_movie = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/ourMovies.csv"
    names_movie = ['ourId', 'movieId','year', 'rating', 'duration','director','actor1','actor2','actor3', 'gross', 'country', 'budget',
             'Mystery', 'Romance',
             'Sci-Fi','Fantasy', 'Horror', 'Film-Noir','Crime', 'Drama', 'Children', 'Animation', 'War', 'Adventure',
             'Action', 'Comedy', 'Documentary', 'Musical', 'Thriller', 'Western']
    movies = pandas.read_csv(url_movie, names=names_movie)


    url2 = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/mojrating.csv"
    names2 = ['userId','movieId', 'rating', 'timestamp']
    ratings_loaded = pandas.read_csv(url2, names=names2)

    n_users = ratings_loaded.userId.unique().shape[0]
    train_data, test_data = model_selection.train_test_split(ratings_loaded, test_size=0.20)
    movie_num = movies.shape[0]
    train_data_matrix = numpy.zeros((n_users, movie_num))

    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]

    test_data_matrix = numpy.zeros((n_users, movie_num))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    predicted = predict(train_data_matrix, user_similarity)
    print 'User-based CF RMSE: ' + str(rmse(predicted, test_data_matrix))
    new_sim = predicted[4]
    rated_indexes = train_data_matrix[4].nonzero()[0]
    rated_movies = []
    ratings_movies = []
    """
    train_data_ind, test_data_ind = model_selection.train_test_split(rated_indexes, test_size=0.20)
    for el in train_data_ind:
        filtered = movies[movies['ourId'] == el]
        for line2 in filtered.itertuples():
            movie_data = [line2[1], line2[2],line2[3], line2[4], line2[5], line2[6], line2[7], line2[8],
                          line2[9], line2[10], line2[11], line2[12], line2[13], line2[14], line2[15], line2[16],
                          line2[17], line2[18], line2[19], line2[20], line2[21], line2[22], line2[23], line2[24],
                          line2[25], line2[26], line2[27], line2[28], line2[29], line2[30]]
            rated_movies.append(movie_data)
            ratings_movies.append(str(train_data_matrix[4, el]))
    """
    scaler = StandardScaler()
    scaler.fit(rated_movies)
    X_train = scaler.transform(rated_movies)

    print("poceo")
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 30), random_state=1)
    clf.fit(X_train, ratings_movies)
    print("zavrsio")
    """
    indexes = numpy.argpartition(new_sim, -30)[-30:]
    print(indexes)
    to_continue = []
    for x in indexes:
        if x not in rated_indexes:
            to_continue.append(x)

    movies_to_predict = []
    real_ratings = []
    for el in test_data_ind:
        filtered = movies[movies['ourId'] == el]
        for line2 in filtered.itertuples():
            movie_data = [line2[1], line2[2],line2[3], line2[4], line2[5], line2[6], line2[7], line2[8],
                          line2[9], line2[10], line2[11], line2[12], line2[13], line2[14], line2[15], line2[16],
                          line2[17], line2[18], line2[19], line2[20], line2[21], line2[22], line2[23], line2[24],
                          line2[25], line2[26], line2[27], line2[28], line2[29], line2[30]]
            movies_to_predict.append(movie_data)
            real_ratings.append(str(train_data_matrix[4, el]))
    scaler.fit(movies_to_predict)
    X_test = scaler.transform(movies_to_predict)

    predictions = clf.predict(X_test)
    print(real_ratings)
    print(predictions)
    print(confusion_matrix(real_ratings,predictions))
    print(classification_report(real_ratings,predictions))
    """