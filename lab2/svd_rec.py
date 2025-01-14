import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""

    # Drop 0 ratings
    ratingDf = df[df['Book-Rating'] <= 0]

    # Drop unpopular books (with ratings amount < 1)
    ratingDf = ratingDf.groupby('ISBN').filter(lambda x: len(x) > 1)

    # Drop inactive users
    ratingDf = ratingDf.groupby('User-ID').filter(lambda x: len(x) > 1)

    return ratingDf.reset_index(drop=True)


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""

    # ...
    svd = SVD()
    # ...
    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)
