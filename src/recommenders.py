import numpy as np
import pandas as pd
# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight
# Для работы с матрицами
from scipy.sparse import csr_matrix


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.data = data.copy()
        self.popularity = self.data.groupby('item_id')['quantity'].sum().reset_index()
        self.popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        self.popularity = self.popularity.sort_values('n_sold', ascending=False).item_id.tolist()

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_data(data, val_lvl_1_size_weeks=6, val_lvl_2_size_weeks=3):
        data_train_lvl_1 = data[data['week_no'] < data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)]
        data_val_lvl_1 = data[
            (data['week_no'] >= data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)) &
            (data['week_no'] < data['week_no'].max() - val_lvl_2_size_weeks)]

        data_train_lvl_2 = data_val_lvl_1.copy()
        data_val_lvl_2 = data[data['week_no'] >= data['week_no'].max() - val_lvl_2_size_weeks]

        return data_train_lvl_1, data_val_lvl_1, data_train_lvl_2, data_val_lvl_2

    @staticmethod
    def prepare_matrix(data):

        # your_code
        user_item_matrix = pd.pivot_table(data=data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0)

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_als_recommendations(self, user, N=5):

        filter_items = [self.itemid_to_id[999999]]
        if user not in self.userid_to_id.keys():
            return self.popularity[:N]

        # your_code
        sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()

        res = [self.id_to_itemid[rec[0]] for rec in
               self.model.recommend(userid=self.userid_to_id[user],
                                    user_items=sparse_user_item,  # на вход user-item matrix
                                    N=N,
                                    filter_already_liked_items=False,
                                    filter_items=filter_items,
                                    recalculate_user=True)]
        if len(res) < N:
            res = np.append(res, self.popularity[:(N - len(res))])

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        if user not in self.userid_to_id.keys():
            return self.popularity[:N]

        filter_items = [self.itemid_to_id[999999]]

        # your_code
        sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()

        res = [self.id_to_itemid[rec[0]] for rec in
               self.own_recommender.recommend(userid=self.userid_to_id[user],
                                              user_items=sparse_user_item,  # на вход user-item matrix
                                              N=N,
                                              filter_already_liked_items=False,
                                              filter_items=filter_items,
                                              recalculate_user=True)]

        if len(res) < N:
            res = np.append(res, self.popularity[:(N - len(res))])

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
