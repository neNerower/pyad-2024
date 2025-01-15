import pickle
import re
import nltk
import pandas as pd
import sklearn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

nltk.download("stopwords")
nltk.download('punkt_tab')


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""

    # Drop image links
    bookDf = df.drop(df.columns[-3:], axis=1)

    # Drop books from future
    bookDf['Year-Of-Publication'] = pd.to_numeric(bookDf['Year-Of-Publication'], errors='coerce')
    bookDf = bookDf[bookDf['Year-Of-Publication'] <= 2016]
    
    # Drop incomplete data
    bookDf = bookDf.dropna()

    return bookDf.reset_index(drop=True)


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""

    # Drop 0 ratings
    ratingDf = df[df['Book-Rating'] > 0]

    # Drop unpopular books (with ratings amount < 1)
    ratingDf = ratingDf.groupby('ISBN').filter(lambda x: len(x) > 1)

    # Drop inactive users
    ratingDf = ratingDf.groupby('User-ID').filter(lambda x: len(x) > 1)

    return ratingDf.reset_index(drop=True)


STOPWORDS = set(stopwords.words("english"))
def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """

    tokens = [token.lower() for token in word_tokenize(text)]
    tokens_no_punkt = [token for token in tokens if str.isalpha(token)]
    tokens_no_punkt_no_stopwords = [token for token in tokens_no_punkt if token not in STOPWORDS]
    return ' '.join(tokens_no_punkt_no_stopwords)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""

    # Clear data
    cleared_books = books_preprocessing(books)
    cleared_books["Book-Title"] = cleared_books["Book-Title"].apply(title_preprocessing)
    cleared_ratings = ratings_preprocessing(ratings)

    # Compute avg book ratings
    avg_book_ratings = cleared_ratings.groupby('ISBN').agg({'Book-Rating': 'mean'})
    rated_books = pd.merge(cleared_books, avg_book_ratings, on='ISBN', how='inner')
    
    X = rated_books[["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]]
    Y = rated_books["Book-Rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    # Prepare pipeline
    preprocessor = ColumnTransformer(
        transformers = [
            ('tf-idf', TfidfVectorizer(), 'Book-Title'),
            ('numbers', StandardScaler(), ['Year-Of-Publication']),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Publisher', 'Book-Author']),
        ]
    )

    reg = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor(max_iter=1000, tol=1e-3))
    ])

    # Train and test model
    reg.fit(X_train, y_train)

    prediction = reg.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    print(f"MAE: {mae}")

    # Save model
    with open("linreg.pkl", "wb") as file:
        pickle.dump(reg, file)
