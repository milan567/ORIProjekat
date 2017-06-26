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

    with open("C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/mojrating.csv", 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                             quoting=csv.QUOTE_MINIMAL)

        for line in ratings.itertuples():
            filtered = movies[(movies['movieId'] == line[2])]
            for line2 in filtered.itertuples():
                spamwriter.writerow([line[1]] + [line2[1]] + [line[3]] + [line[4]])


def get_movie_num():
    url = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/mojfilm.csv"
    names = ['ourId','movieId', 'title', 'genres']
    movies = pandas.read_csv(url, names=names)
    return movies.shape[0]

if __name__=='__main__':
    url2 = "C:/Users/Nikolina/Desktop/ori/ml-latest-small/ml-latest-small/mojrating.csv"
    names2 = ['userId','movieId', 'rating', 'timestamp']
    ratings = pandas.read_csv(url2, names=names2)

    n_users = ratings.userId.unique().shape[0]
    train_data, test_data = model_selection.train_test_split(ratings, test_size=0.20)
    movie_num = get_movie_num()
    train_data_matrix = numpy.zeros((n_users, movie_num))

    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]

    test_data_matrix = numpy.zeros((n_users, movie_num))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]

    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
    print(max(item_similarity[0]))