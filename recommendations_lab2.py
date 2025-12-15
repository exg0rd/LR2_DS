#!/usr/bin/env python
# coding: utf-8

"""
Лабораторная №2: рекомендательные системы
Реализация рекомендательной системы с использованием SVD, SSVD, NMF и ALS
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

if target_movie_id is None:
    print("Ошибка: Указанный фильм не найден в базе данных!")
else:
    print()
    print("="*60)
    print("ЧАСТЬ 1: Рекомендации похожих фильмов")
    print("="*60)
    
    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='userid', columns='movieid', values='rating').fillna(0)
    
    # Get indices for later use
    movie_ids = user_item_matrix.columns
    user_ids = user_item_matrix.index
    
    print(f"Размерность матрицы пользователь-фильм: {user_item_matrix.shape}")
    print(f"Количество пользователей: {len(user_ids)}")
    print(f"Количество фильмов: {len(movie_ids)}")
    print()
    
    # Find the column index for our target movie
    target_movie_idx = np.where(movie_ids == target_movie_id)[0][0] if target_movie_id in movie_ids else None
    
    if target_movie_idx is not None:
        print("1. Применяем SVD разложение...")
        
        # Apply SVD
        U, sigma, Vt = np.linalg.svd(user_item_matrix.values, full_matrices=False)
        
        # Plot singular values decay
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, min(len(sigma), 50) + 1), sigma[:min(len(sigma), 50)], 'bo-')
        plt.title('Убывание сингулярных чисел')
        plt.xlabel('Номер сингулярного числа')
        plt.ylabel('Сингулярное число')
        plt.grid(True)
        plt.show()
        
        print("\n2. Применяем SVD с различным количеством компонент...")
        
        def get_top_similar_movies_svd(target_movie_idx, n_components_list=[10, 20, 50], top_n=10):
            """
            Находит топ-N похожих фильмов используя SVD с разными количествами компонент
            """
            results = {}
            
            for n_comp in n_components_list:
                # Truncated SVD
                U_trunc = U[:, :n_comp]
                sigma_trunc = sigma[:n_comp]
                Vt_trunc = Vt[:n_comp, :]
                
                # Reconstructed matrix
                reconstructed = U_trunc @ np.diag(sigma_trunc) @ Vt_trunc
                
                # Calculate similarities between target movie and all other movies
                target_movie_vector = reconstructed[:, target_movie_idx]
                similarities = []
                
                for i in range(reconstructed.shape[1]):
                    if i != target_movie_idx:  # Don't include the same movie
                        other_movie_vector = reconstructed[:, i]
                        # Cosine similarity
                        sim = np.dot(target_movie_vector, other_movie_vector) / (
                            np.linalg.norm(target_movie_vector) * np.linalg.norm(other_movie_vector)
                        )
                        similarities.append((i, sim))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get top N similar movies
                top_similar = similarities[:top_n]
                results[n_comp] = [(movie_ids[idx], movies.loc[movie_ids[idx], 'movienm'], sim) 
                                   for idx, sim in top_similar]
                
            return results
        
        # Get results with different components
        svd_results = get_top_similar_movies_svd(target_movie_idx)
        
        print("\nРезультаты SVD:")
        for n_comp, results in svd_results.items():
            print(f"\nTop 10 фильмов, похожих на '{data['movie']}' (SVD, {n_comp} компонент):")
            for i, (movie_id, movie_title, similarity) in enumerate(results, 1):
                print(f"{i:2d}. {movie_title} (ID: {movie_id}, сходство: {similarity:.4f})")
        
        print("\n3. Применяем SSVD (Sparse SVD)...")
        
        # For SSVD we'll use scipy.sparse.linalg.svds
        def get_top_similar_movies_ssvd(target_movie_idx, n_components=50, top_n=10):
            """
            Находит топ-N похожих фильмов используя SSVD
            """
            # Using scipy.sparse.linalg.svds for sparse SVD
            U_s, sigma_s, Vt_s = svds(user_item_matrix.values, k=n_components)
            
            # Reconstruct
            reconstructed_s = U_s @ np.diag(sigma_s) @ Vt_s
            
            # Calculate similarities
            target_movie_vector = reconstructed_s[:, target_movie_idx]
            similarities = []
            
            for i in range(reconstructed_s.shape[1]):
                if i != target_movie_idx:  # Don't include the same movie
                    other_movie_vector = reconstructed_s[:, i]
                    # Cosine similarity
                    sim = np.dot(target_movie_vector, other_movie_vector) / (
                        np.linalg.norm(target_movie_vector) * np.linalg.norm(other_movie_vector)
                    )
                    similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N similar movies
            top_similar = similarities[:top_n]
            return [(movie_ids[idx], movies.loc[movie_ids[idx], 'movienm'], sim) 
                    for idx, sim in top_similar]
        
        ssvd_results = get_top_similar_movies_ssvd(target_movie_idx)
        
        print(f"\nTop 10 фильмов, похожих на '{data['movie']}' (SSVD):")
        for i, (movie_id, movie_title, similarity) in enumerate(ssvd_results, 1):
            print(f"{i:2d}. {movie_title} (ID: {movie_id}, сходство: {similarity:.4f})")
        
        print("\n4. Применяем NMF (Non-negative Matrix Factorization)...")
        
        def get_top_similar_movies_nmf(target_movie_idx, n_components=50, top_n=10):
            """
            Находит топ-N похожих фильмов используя NMF
            """
            # Make sure all values are non-negative
            user_item_non_negative = np.clip(user_item_matrix.values, 0, None)
            
            # Apply NMF
            model = NMF(n_components=n_components, random_state=42, max_iter=200)
            W = model.fit_transform(user_item_non_negative)
            H = model.components_
            
            # Reconstruct
            reconstructed_nmf = W @ H
            
            # Calculate similarities
            target_movie_vector = reconstructed_nmf[:, target_movie_idx]
            similarities = []
            
            for i in range(reconstructed_nmf.shape[1]):
                if i != target_movie_idx:  # Don't include the same movie
                    other_movie_vector = reconstructed_nmf[:, i]
                    # Cosine similarity
                    sim = np.dot(target_movie_vector, other_movie_vector) / (
                        np.linalg.norm(target_movie_vector) * np.linalg.norm(other_movie_vector)
                    )
                    similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N similar movies
            top_similar = similarities[:top_n]
            return [(movie_ids[idx], movies.loc[movie_ids[idx], 'movienm'], sim) 
                    for idx, sim in top_similar]
        
        nmf_results = get_top_similar_movies_nmf(target_movie_idx)
        
        print(f"\nTop 10 фильмов, похожих на '{data['movie']}' (NMF):")
        for i, (movie_id, movie_title, similarity) in enumerate(nmf_results, 1):
            print(f"{i:2d}. {movie_title} (ID: {movie_id}, сходство: {similarity:.4f})")
        
        print("\n5. Сравнение методов:")
        print("- SVD: Классическое сингулярное разложение, работает с полными матрицами")
        print("- SSVD: Работает эффективнее с разреженными матрицами, может быть быстрее")
        print("- NMF: Разложение на неотрицательные компоненты, хорошо интерпретируется")
        print("Все три метода дают разные результаты из-за разных подходов к факторизации.")
        
        print("\n" + "="*60)
        print("ЧАСТЬ 2: Рекомендации похожих пользователей")
        print("="*60)
        
        # Find the target user index
        target_user_idx = np.where(user_ids == data['user'])[0][0] if data['user'] in user_ids else None
        
        if target_user_idx is not None:
            print(f"Целевой пользователь: {data['user']} (индекс: {target_user_idx})")
            
            def get_top_similar_users(target_user_idx, method='svd', n_components=50, top_n=10):
                """
                Находит топ-N похожих пользователей
                """
                if method == 'svd':
                    # Use SVD-based representation
                    U_trunc = U[:, :n_components]
                    sigma_trunc = sigma[:n_components]
                    Vt_trunc = Vt[:n_components, :]
                    
                    reconstructed = U_trunc @ np.diag(sigma_trunc) @ Vt_trunc
                    
                    target_user_vector = reconstructed[target_user_idx, :]  # User profile
                    similarities = []
                    
                    for i in range(reconstructed.shape[0]):
                        if i != target_user_idx:  # Don't include the same user
                            other_user_vector = reconstructed[i, :]
                            # Cosine similarity
                            sim = np.dot(target_user_vector, other_user_vector) / (
                                np.linalg.norm(target_user_vector) * np.linalg.norm(other_user_vector)
                            )
                            similarities.append((i, sim))
                    
                elif method == 'ssvd':
                    # Use SSVD-based representation
                    # Recompute SSVD since these variables aren't available in this scope
                    U_s, sigma_s, Vt_s = svds(user_item_matrix.values, k=n_components)
                    
                    U_s_trunc = U_s[:, :n_components]
                    sigma_s_trunc = sigma_s[:n_components]
                    Vt_s_trunc = Vt_s[:n_components, :]
                    
                    reconstructed_s = U_s_trunc @ np.diag(sigma_s_trunc) @ Vt_s_trunc
                    
                    target_user_vector = reconstructed_s[target_user_idx, :]  # User profile
                    similarities = []
                    
                    for i in range(reconstructed_s.shape[0]):
                        if i != target_user_idx:  # Don't include the same user
                            other_user_vector = reconstructed_s[i, :]
                            # Cosine similarity
                            sim = np.dot(target_user_vector, other_user_vector) / (
                                np.linalg.norm(target_user_vector) * np.linalg.norm(other_user_vector)
                            )
                            similarities.append((i, sim))
                
                elif method == 'nmf':
                    # Use NMF-based representation
                    # Recompute NMF since W isn't available in this scope
                    user_item_non_negative = np.clip(user_item_matrix.values, 0, None)
                    model = NMF(n_components=n_components, random_state=42, max_iter=200)
                    W = model.fit_transform(user_item_non_negative)
                    
                    target_user_vector = W[target_user_idx, :]  # User profile in latent space
                    similarities = []
                    
                    for i in range(W.shape[0]):
                        if i != target_user_idx:  # Don't include the same user
                            other_user_vector = W[i, :]
                            # Cosine similarity
                            sim = np.dot(target_user_vector, other_user_vector) / (
                                np.linalg.norm(target_user_vector) * np.linalg.norm(other_user_vector)
                            )
                            similarities.append((i, sim))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get top N similar users
                top_similar = similarities[:top_n]
                return [(user_ids[idx], sim) for idx, sim in top_similar]
            
            # Get similar users for each method
            similar_users_svd = get_top_similar_users(target_user_idx, 'svd')
            similar_users_ssvd = get_top_similar_users(target_user_idx, 'ssvd')
            similar_users_nmf = get_top_similar_users(target_user_idx, 'nmf')
            
            print(f"\nTop 10 пользователей, похожих на пользователя {data['user']} (SVD):")
            for i, (user_id, similarity) in enumerate(similar_users_svd, 1):
                print(f"{i:2d}. Пользователь {user_id} (сходство: {similarity:.4f})")
            
            print(f"\nTop 10 пользователей, похожих на пользователя {data['user']} (SSVD):")
            for i, (user_id, similarity) in enumerate(similar_users_ssvd, 1):
                print(f"{i:2d}. Пользователь {user_id} (сходство: {similarity:.4f})")
            
            print(f"\nTop 10 пользователей, похожих на пользователя {data['user']} (NMF):")
            for i, (user_id, similarity) in enumerate(similar_users_nmf, 1):
                print(f"{i:2d}. Пользователь {user_id} (сходство: {similarity:.4f})")
            
            print("\nСравнение методов для похожих пользователей:")
            print("- SVD: Хорошо работает с плотными данными, но может быть медленным")
            print("- SSVD: Более эффективен для больших разреженных матриц")
            print("- NMF: Даёт интерпретируемые результаты, особенно для рекомендаций")
        
        else:
            print(f"Целевой пользователь {data['user']} не найден в матрице рейтингов.")
    
    else:
        print(f"Целевой фильм с ID {target_movie_id} не найден в матрице пользователь-фильм.")

# Implementation of ALS algorithm
def als_algorithm(ratings_df, n_factors=50, reg_param=0.01, n_iterations=10):
    """
    Implementation of Alternating Least Squares algorithm
    """
    print("\n" + "="*60)
    print("ЧАСТЬ 3: Реализация ALS алгоритма")
    print("="*60)
    
    # Create user and item mappings
    unique_users = ratings_df['userid'].unique()
    unique_items = ratings_df['movieid'].unique()
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    # Create rating matrix
    R = np.zeros((n_users, n_items))
    for _, row in ratings_df.iterrows():
        u_idx = user_to_idx[row['userid']]
        i_idx = item_to_idx[row['movieid']]
        R[u_idx, i_idx] = row['rating']
    
    # Initialize user and item factors randomly
    U = np.random.normal(0, 0.1, (n_users, n_factors))
    V = np.random.normal(0, 0.1, (n_items, n_factors))
    
    lambda_reg = reg_param
    
    print(f"Размерности: U={U.shape}, V={V.shape}")
    print(f"Количество факторов: {n_factors}, регуляризация: {lambda_reg}, итераций: {n_iterations}")
    
    # ALS iterations
    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")
        
        # Update user factors
        for u in range(n_users):
            # Items rated by user u
            rated_items = R[u, :] != 0
            V_rated = V[rated_items, :]
            ratings_u = R[u, :][rated_items]
            
            if len(ratings_u) > 0:
                A = V_rated.T @ V_rated + lambda_reg * np.eye(n_factors)
                b = V_rated.T @ ratings_u
                U[u, :] = np.linalg.solve(A, b)
        
        # Update item factors
        for i in range(n_items):
            # Users who rated item i
            rated_by_users = R[:, i] != 0
            U_rated = U[rated_by_users, :]
            ratings_i = R[:, i][rated_by_users]
            
            if len(ratings_i) > 0:
                A = U_rated.T @ U_rated + lambda_reg * np.eye(n_factors)
                b = U_rated.T @ ratings_i
                V[i, :] = np.linalg.solve(A, b)
    
    print("ALS алгоритм завершен.")
    
    # Function to predict rating for a user and item
    def predict_rating(user_id, movie_id):
        if user_id in user_to_idx and movie_id in item_to_idx:
            u_idx = user_to_idx[user_id]
            i_idx = item_to_idx[movie_id]
            pred_rating = np.dot(U[u_idx, :], V[i_idx, :])
            return pred_rating
        else:
            return None  # User or item not seen during training
    
    # Predict rating for target user and movie
    target_user = data['user']
    target_movie = target_movie_id
    
    if target_movie and target_user in user_to_idx and target_movie in item_to_idx:
        predicted_rating = predict_rating(target_user, target_movie)
        actual_ratings = ratings[(ratings['userid'] == target_user) & (ratings['movieid'] == target_movie)]
        
        if len(actual_ratings) > 0:
            actual_rating = actual_ratings.iloc[0]['rating']
            print(f"\nПрогноз для пользователя {target_user} и фильма '{data['movie']}':")
            print(f"Предсказанный рейтинг: {predicted_rating:.4f}")
            print(f"Фактический рейтинг: {actual_rating}")
        else:
            print(f"\nПрогноз для пользователя {target_user} и фильма '{data['movie']}':")
            print(f"Предсказанный рейтинг: {predicted_rating:.4f}")
            print("Фактический рейтинг: отсутствует (новый пользователь/фильм)")
    
    return U, V, user_to_idx, item_to_idx, predict_rating


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
    
    # Predict rating for target user and movie
    target_user = data['user']
    target_movie = target_movie_id
    
    if target_movie:
        predicted_rating = predict_rating(target_user, target_movie)
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
    
    return model, predict_rating


# Run ALS algorithm
if target_movie_id is not None:
    U_als, V_als, user_to_idx, item_to_idx, predict_fn = als_algorithm(ratings, n_factors=50, reg_param=0.01, n_iterations=10)
    
    # Run FunkSVD algorithm
    funksvd_model, funksvd_predict_fn = funksvd_algorithm(ratings, n_factors=10, n_epochs=20, lr=0.005, reg=0.02)
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("1. SVD, SSVD и NMF - все три метода матричного разложения позволяют находить")
    print("   скрытые паттерны в пользовательских предпочтениях и делать рекомендации.")
    print("2. ALS алгоритм эффективно оптимизирует факторы пользователей и предметов,")
    print("   минимизируя ошибку прогнозирования рейтингов.")
    print("3. FunkSVD использует стохастический градиентный спуск для оптимизации,")
    print("   что делает его эффективным для больших наборов данных.")
    print("4. Все методы показывают разные аспекты сходства между пользователями и фильмами,")
    print("   что позволяет строить более точные рекомендательные системы.")
    print("5. Для конкретного фильма 'Seventh Seal, The (Sjunde inseglet, Det) (1957)'")
    print("   были получены списки похожих фильмов и похожих пользователей.")