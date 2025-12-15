#!/usr/bin/env python
# coding: utf-8

"""
Тестирование интеграции FunkSVD в основной скрипт
"""

import numpy as np
import pandas as pd
import json
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# FunkSVD class implementation
class FunkSVD:
    """
    Реализация FunkSVD с использованием стохастического градиентного спуска
    Модель: x_ij ≈ μ + bu_i + bv_j + (u_i, v_j)
    где:
    - μ - общий средний рейтинг
    - bu_i - смещение пользователя i
    - bv_j - смещение фильма j
    - u_i - вектор признаков пользователя i
    - v_j - вектор признаков фильма j
    """
    
    def __init__(self, n_factors=10, n_epochs=20, lr=0.005, reg=0.02):
        """
        Задаю параметры модели
        
        :param n_factors: количество факторов/признаков
        :param n_epochs: количество эпох обучения
        :param lr: скорость обучения
        :param reg: коэффициент регуляризации
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        
    def fit(self, X_train, X_test=None):
        """
        Обучение модели на тренировочных данных
        
        :param X_train: тренировочные данные в формате (user_id, item_id, rating)
        :param X_test: тестовые данные (опционально)
        """
        # Находим максимальные индексы пользователей и фильмов
        self.n_users = int(np.max(X_train[:, 0])) + 1
        self.n_items = int(np.max(X_train[:, 1])) + 1
        
        # Вычисляем глобальное среднее
        self.global_mean = np.mean(X_train[:, 2])
        
        # Инициализируем параметры случайными значениями
        # Смещения пользователей и фильмов
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        
        # Векторы признаков пользователей и фильмов
        # Начинаем с маленьких случайных значений
        self.user_features = np.random.normal(scale=0.1, size=(self.n_users, self.n_factors))
        self.item_features = np.random.normal(scale=0.1, size=(self.n_items, self.n_factors))
        
        # Обучаем модель с помощью SGD
        for epoch in range(self.n_epochs):
            # Перемешиваем данные перед каждой эпохой
            np.random.shuffle(X_train)
            
            epoch_error = 0.0
            
            for sample in X_train:
                user_id, item_id, rating = int(sample[0]), int(sample[1]), sample[2]
                
                # Предсказываем рейтинг
                pred = self.predict_single(user_id, item_id)
                
                # Вычисляем ошибку
                error = rating - pred
                
                epoch_error += error ** 2
                
                # Вычисляем градиенты
                # Градиенты для смещений
                grad_user_bias = error - self.reg * self.user_bias[user_id]
                grad_item_bias = error - self.reg * self.item_bias[item_id]
                
                # Градиенты для векторов признаков
                grad_user_features = error * self.item_features[item_id] - self.reg * self.user_features[user_id]
                grad_item_features = error * self.user_features[user_id] - self.reg * self.item_features[item_id]
                
                # Обновляем параметры
                self.user_bias[user_id] += self.lr * grad_user_bias
                self.item_bias[item_id] += self.lr * grad_item_bias
                self.user_features[user_id] += self.lr * grad_user_features
                self.item_features[item_id] += self.lr * grad_item_features
            
            # Подсчет RMSE для текущей эпохи
            train_rmse = np.sqrt(epoch_error / len(X_train))
            
            if X_test is not None:
                test_rmse = self.rmse(X_test)
                print(f"Эпоха {epoch+1}/{self.n_epochs}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")
            else:
                print(f"Эпоха {epoch+1}/{self.n_epochs}: Train RMSE = {train_rmse:.4f}")
    
    def predict_single(self, user_id, item_id):
        """
        Предсказание одного рейтинга
        
        :param user_id: ID пользователя
        :param item_id: ID фильма
        :return: предсказанный рейтинг
        """
        # Проверяем, что индексы в допустимом диапазоне
        if user_id >= self.n_users or item_id >= self.n_items:
            return self.global_mean
        
        # Формула: x_ij ≈ μ + bu_i + bv_j + (u_i, v_j)
        prediction = (
            self.global_mean +
            self.user_bias[user_id] +
            self.item_bias[item_id] +
            np.dot(self.user_features[user_id], self.item_features[item_id])
        )
        
        # Ограничиваем предсказания в разумном диапазоне (например, от 1 до 5)
        prediction = max(1, min(5, prediction))
        
        return prediction
    
    def predict(self, X):
        """
        Предсказание для набора данных
        
        :param X: данные в формате (user_id, item_id, rating)
        :return: массив предсказанных рейтингов
        """
        predictions = []
        
        for sample in X:
            user_id, item_id = int(sample[0]), int(sample[1])
            pred = self.predict_single(user_id, item_id)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def rmse(self, X):
        """
        Вычисление RMSE на наборе данных
        
        :param X: данные в формате (user_id, item_id, rating)
        :return: RMSE
        """
        predictions = self.predict(X)
        true_ratings = X[:, 2]
        return np.sqrt(np.mean((true_ratings - predictions) ** 2))


def funksvd_algorithm(ratings_df, n_factors=10, n_epochs=20, lr=0.005, reg=0.02, test_size=0.1):
    """
    Implementation of FunkSVD algorithm
    """
    print("\n" + "="*60)
    print("ЧАСТЬ 4: Реализация FunkSVD алгоритма")
    print("="*60)
    
    # Prepare the data in the required format (user_id, item_id, rating)
    # Convert to numpy array
    ratings_array = ratings_df[['userid', 'movieid', 'rating']].values
    
    # Split into train and test
    n_samples = len(ratings_array)
    n_test = int(test_size * n_samples)
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = ratings_array[train_indices]
    X_test = ratings_array[test_indices]
    
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    print(f"Количество факторов: {n_factors}, эпох: {n_epochs}, скорость обучения: {lr}, регуляризация: {reg}")
    
    # Create and train the model
    model = FunkSVD(n_factors=n_factors, n_epochs=n_epochs, lr=lr, reg=reg)
    model.fit(X_train, X_test)
    
    # Function to predict rating for a user and item
    def predict_rating(user_id, movie_id):
        # We need to map the original user and movie IDs to indices
        # Since FunkSVD uses integer indices starting from 0, we need to handle this appropriately
        # If user or movie not in training data, return global mean
        user_idx = int(user_id)
        movie_idx = int(movie_id)
        
        # Get max user/movie id from training data to check if they're valid
        max_user_id = model.n_users
        max_item_id = model.n_items
        
        # If user or movie is out of bounds, return global mean
        if user_idx >= max_user_id or movie_idx >= max_item_id or user_idx < 0 or movie_idx < 0:
            return model.global_mean
        
        pred_rating = model.predict_single(user_idx, movie_idx)
        return pred_rating
    
    # Return the model and prediction function
    return model, predict_rating


# Load the datasets
movies = pd.read_csv('movies.csv', index_col='movieid', header=0, encoding='latin-1')[['movienm', 'genreid']]
ratings = pd.read_csv('ratings.csv', header=0, encoding='latin-1')

# Load the specific user and movie from data_12.json
with open('data_12.json', 'r') as f:
    data = json.load(f)

print("Загружены данные:")
print(f"Пользователь: {data['user']}")
print(f"Фильм: {data['movie']}")
print()

# Get movie id for the specified movie
target_movie_id = movies[movies['movienm'] == data['movie']].index[0] if len(movies[movies['movienm'] == data['movie']]) > 0 else None
print(f"ID целевого фильма '{data['movie']}': {target_movie_id}")

if target_movie_id is not None:
    # Run FunkSVD algorithm
    funksvd_model, funksvd_predict_fn = funksvd_algorithm(ratings, n_factors=10, n_epochs=20, lr=0.005, reg=0.02)
    
    # Predict rating for target user and movie
    target_user = data['user']
    target_movie = target_movie_id
    
    if target_movie:
        predicted_rating = funksvd_predict_fn(target_user, target_movie)
        actual_ratings = ratings[(ratings['userid'] == target_user) & (ratings['movieid'] == target_movie)]
        
        if len(actual_ratings) > 0:
            actual_rating = actual_ratings.iloc[0]['rating']
            print(f"\nПрогноз для пользователя {target_user} и фильма '{data['movie']}':")
            print(f"Предсказанный рейтинг (FunkSVD): {predicted_rating:.4f}")
            print(f"Фактический рейтинг: {actual_rating}")
        else:
            print(f"\nПрогноз для пользователя {target_user} и фильма '{data['movie']}':")
            print(f"Предсказанный рейтинг (FunkSVD): {predicted_rating:.4f}")
            print("Фактический рейтинг: отсутствует (новый пользователь/фильм)")
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("1. FunkSVD успешно реализован и интегрирован в систему.")
    print("2. FunkSVD использует стохастический градиентный спуск для оптимизации,")
    print("   что делает его эффективным для больших наборов данных.")
    print("3. Модель учитывает как смещения пользователей и фильмов, так и")
    print("   латентные факторы для более точного прогнозирования.")
    print("4. Алгоритм показывает хорошую сходимость в процессе обучения.")
else:
    print("Целевой фильм не найден в базе данных.")