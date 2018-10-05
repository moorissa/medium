# Formation data challenge
# Moorissa Tjokro

# This is an additional python scripts I used for developing recommendation model
# Notebook uses the turicreate library for a more efficient evaluation approach

import pandas as pd
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine

class Formation:
    def __init__(self, input_file, customer_file, output_file, csv_output=True):

        self.input_file = input_file
        self.customer_file = customer_file
        self.output_file = output_file
        self.csv_output = True
        self.n_recommendations = 10
        self.n_neighbors = 10


    def load_data(self, filename):
        """
        Loads input data in csv format

        Args:
            filename (str): a csv file, e.g. 'data.csv'

        Returns:
            (pandas.DataFrame)

        """
        return pd.read_csv('../data/'+filename)


    def purchase_frequency(self, user):
        """
        Returns a dictionary of items bought for each given user,
        with each item id as keys and their corresponding number of times
        the item is bought as values.

        Args:
            user (int): user ID

        Returns:
            (dictionary)
        """
        N = self.transactions.set_index('customerId')['products'][user][:]
        if type(N)!=str:
            N = np.array("|".join(N.values.reshape(1, len(N))[0]))
        bought = [int(i) for i in str(N).split('|')]
        return dict(Counter(bought))


    def create_matrix_user_items(self):
        """
        Creates a user-to-item matrix, where index values represent unique
        user IDs and columns represent unique item IDs.
        The matrix shape is (n_users x n_items).

        Args:
            None

        Returns:
            None
        """
        dic_users = {}
        for user in self.transactions.customerId.unique():
            dic_users[user] = self.purchase_frequency(user)

        self.matrix_user_items = np.array(pd.DataFrame(dic_users).T.fillna(0))


    def create_matrix_items(self):
        """
        Creates an item-to-item matrix, where both indices and columns represent
        unique item IDs.
        The matrix shape is square (n_items x n_items).

        Args:
            None

        Returns:
            None
        """
        n = self.n_items
        self.matrix_items = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                self.matrix_items[i][j] = 1-cosine(self.matrix_user_items[:, i],
                                                   self.matrix_user_items[:, j])

    def create_matrix_neighbors(self):
        """
        Creates a matrix for selecting top neighbors for each item i based on
        similarity scores, where index values represent unique items and columns
        represent items that are most similar to that item.
        The matrix shape is square (n_items x n_neighbors).

        Args:
            None

        Returns:
            None
        """
        n = self.n_items
        m = self.n_neighbors

        self.matrix_neighbor_items = np.zeros([n, m])
        self.matrix_neighbor_indices = np.zeros([n, m])

        for i in range(n):
            sorted_indices = np.argsort(self.matrix_items[i])[::-1][:m]
            self.matrix_neighbor_indices[i] = sorted_indices
            self.matrix_neighbor_items[i] = self.matrix_items[i][sorted_indices]

    def create_matrix_similarity(self):
        """
        Creates a similarity matrix, where index values represent unique
        user IDs and columns represent unique item IDs. Scores are filled in
        based on user purchase and neighboring items.
        The matrix shape is (n_users x n_items).

        Args:
            None

        Returns:
            None
        """
        self.matrix_pred = np.zeros([self.n_users, self.n_items])

        for user in range(self.n_users):
            for item in range(self.n_items):
                top_neighbor_item_scores = self.matrix_neighbor_items[item][1:]
                indices = self.matrix_neighbor_indices[item][1:].astype(np.int64)
                user_purchase = self.matrix_user_items[user][indices]
                self.matrix_pred[user][item] = sum(user_purchase*top_neighbor_item_scores)/ \
                                               sum(top_neighbor_item_scores)

    def create_recommendations(self, csv_output=True):
        """
        Creates a recommendation matrix for all items and a dataframe consisting of
        top 10 recommended items. Allows for returning a csv output file with
        specified customerId and their recommendations.

        Args:
            None

        Returns:
            None
        """
        u = self.n_users
        r = self.n_recommendations
        c = 'customerId'
        matrix_recom_scores = np.zeros([u, r])
        matrix_recom_indices = np.zeros([u, r])

        for user in range(u):
            sorted_indices = np.argsort(self.matrix_pred[user])[::-1][:r]
            matrix_recom_indices[user] = sorted_indices
            matrix_recom_scores[user] = self.matrix_pred[user][sorted_indices]

        df_recommend = pd.DataFrame(matrix_recom_indices)
        df_recommend[c] = df_matrix.index
        self.df_top10 = df_recommend[[c]+list(df_recommend.columns[:r])] \
            .astype(np.int64).set_index(c).loc[customers[c]]
        self.df_top10['recommendedProducts'] = self.df_top10[list(range(n_recommendations))] \
            .apply(lambda x: '|'.join(x.fillna('').map(str)), axis=1)
        if csv_output:
            self.df_top10[['recommendedProducts']].to_csv('../output/'+output_file)


    def process_data(self):
        """
        Runs all stages of data processing, from loading the data, matrix transformation,
        evaluating the model, and outputting the recommendation items for users.

        Args:
            None

        Returns:
            None
        """
        # 1. load data
        self.transactions = self.load_data(self.input_file)
        self.customers = self.load_data(self.customer_file)
        self.n_users = self.transactions.customerId.nunique()

        # 2. create user-to-item matrix
        self.create_matrix_user_items()
        self.n_items = self.matrix_user_items.shape[1]

        # 3. create item-to-item matrix
        self.create_matrix_items()

        # 4. create neighboring items matrix
        self.create_matrix_neighbors()

        # 5. create similarity matrix
        self.create_matrix_similarity()

        # 6. create recommendations matrix
        self.create_recommendations(csv_output=self.csv_output)
