__author__ = 'Nikolina'

import pandas
import numpy
from sklearn import model_selection
from sklearn.metrics.pairwise import pairwise_distances
import csv


def repair_data():
    url = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/mojfilm.csv"
    names = ['ourId','movieId', 'title', 'genres']
    movies = pandas.read_csv(url, names=names)

    url2 = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/ratings.csv"
    names2 = ['userId','movieId', 'rating', 'timestamp']
    ratings = pandas.read_csv(url2, names=names2)

    with open("C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/test.csv", 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                             quoting=csv.QUOTE_MINIMAL)

        for line in ratings.itertuples():
            filtered = movies[(movies['movieId'] == line[2])]
            for line2 in filtered.itertuples():
                spamwriter.writerow([line[1]] + [line2[1]] + [line[3]] + [line[4]])
            if line[0] == 100:
                break


def get_movie_num():
    url = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/mojfilm.csv"
    names = ['ourId','movieId', 'title', 'genres']
    movies = pandas.read_csv(url, names=names)
    return movies.shape[0]


def predict(ratings, similarity):
    rated_num = numpy.count_nonzero(ratings, axis=1)
    rated_num[rated_num == 0] = 6
    sum_user_rating = ratings.sum(axis=1)
    mean = sum_user_rating[:, numpy.newaxis]/rated_num[:, numpy.newaxis]
    ratings_diff = (ratings - mean)
    new_sim = similarity.copy()
    new_sim[new_sim > 0.5] = 0.00001
    predicted = mean + new_sim.dot(ratings_diff) / numpy.array([numpy.abs(new_sim).sum(axis=1)]).T
    return predicted

if __name__=='__main__':
    url2 = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/mojrating.csv"
    names2 = ['userId','movieId', 'rating', 'timestamp']
    ratings_loaded = pandas.read_csv(url2, names=names2)

    n_users = ratings_loaded.userId.unique().shape[0]
    train_data, test_data = model_selection.train_test_split(ratings_loaded, test_size=0.20)
    movie_num = get_movie_num()
    train_data_matrix = numpy.zeros((n_users, movie_num))

    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]

    test_data_matrix = numpy.zeros((n_users, movie_num))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    predicted = predict(train_data_matrix, user_similarity)
    new_sim = predicted[4]
    rated_indexes = train_data_matrix[4].nonzero()[0]
    print(rated_indexes)
    indexes = numpy.argpartition(new_sim, -30)[-30:]
    print(indexes)
    to_continue = []
    for x in indexes:
        if x not in rated_indexes:
            to_continue.append(x)