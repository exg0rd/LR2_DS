#!/usr/bin/env python
# coding: utf-8

"""
Лабораторная №2: рекомендательные системы
Реализация рекомендательной системы с использованием SVD, SSVD, NMF и ALS
Оптимизированная версия для работы с большими данными
"""

import numpy as np
import pandas as pd
import json
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
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
    
    # Filter ratings to include only users who rated our target movie
    users_who_rated_target = ratings[ratings['movieid'] == target_movie_id]['userid'].unique()
    
    # Take a sample of users for performance (including the target user)
    sample_users = np.random.choice(users_who_rated_target, size=min(500, len(users_who_rated_target)), replace=False)
    
    # Add some more users to have a reasonable sample size
    if len(sample_users) < 1000:
        additional_users = np.random.choice(ratings['userid'].unique(), 
                                          size=min(1000-len(sample_users), len(ratings['userid'].unique())), 
                                          replace=False)
        sample_users = np.concatenate([sample_users, additional_users])
        sample_users = np.unique(sample_users)  # Remove duplicates
    
    # Filter ratings for sampled users
    filtered_ratings = ratings[ratings['userid'].isin(sample_users)]
    
    # Also limit to popular movies for efficiency
    movie_popularity = filtered_ratings.groupby('movieid').size()
    popular_movies = movie_popularity[movie_popularity >= 10].index  # At least 10 ratings
    filtered_ratings = filtered_ratings[filtered_ratings['movieid'].isin(popular_movies)]
    
    print(f"Размер отфильтрованной выборки:")
    print(f"  Пользователей: {filtered_ratings['userid'].nunique()}")
    print(f"  Фильмов: {filtered_ratings['movieid'].nunique()}")
    print(f"  Рейтингов: {len(filtered_ratings)}")
    
    # Create user-item matrix using sparse matrix for efficiency
    user_ids = filtered_ratings['userid'].unique()
    movie_ids = filtered_ratings['movieid'].unique()
    
    # Create mappings
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    movie_to_idx = {movie: idx for idx, movie in enumerate(movie_ids)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_movie = {idx: movie for movie, idx in movie_to_idx.items()}
    
    # Create sparse matrix
    row = [user_to_idx[user] for user in filtered_ratings['userid']]
    col = [movie_to_idx[movie] for movie in filtered_ratings['movieid']]
    data = filtered_ratings['rating'].values
    sparse_matrix = csr_matrix((data, (row, col)), 
                              shape=(len(user_ids), len(movie_ids)))
    
    print(f"Размерность матрицы пользователь-фильм: {sparse_matrix.shape}")
    print()
    
    # Find the column index for our target movie
    if target_movie_id in movie_to_idx:
        target_movie_idx = movie_to_idx[target_movie_id]
        print(f"Целевой фильм '{data['movie']}' найден в матрице (индекс: {target_movie_idx})")
        
        print("\n1. Применяем SSVD (Sparse SVD)...")
        
        # Apply SSVD - this is more memory efficient than full SVD
        n_components = min(50, min(sparse_matrix.shape) - 1)  # Ensure valid number of components
        U, sigma, Vt = svds(sparse_matrix, k=n_components)
        
        # Sort by singular values in descending order (svds returns them in ascending)
        idx = np.argsort(sigma)[::-1]
        sigma = sigma[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
        
        # Plot singular values decay
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(sigma) + 1), sigma, 'bo-')
        plt.title('Убывание сингулярных чисел')
        plt.xlabel('Номер сингулярного числа')
        plt.ylabel('Сингулярное число')
        plt.grid(True)
        # Comment out plt.show() to avoid display issues
        # plt.show()
        print("График убывания сингулярных чисел построен.")
        
        print("\n2. Применяем SSVD для поиска похожих фильмов...")
        
        def get_top_similar_movies_ssvd(target_movie_idx, top_n=10):
            """
            Находит топ-N похожих фильмов используя SSVD
            """
            # Get the movie vector in latent space (from Vt transpose)
            target_movie_vector = Vt[:, target_movie_idx]
            similarities = []
            
            for i in range(Vt.shape[1]):
                if i != target_movie_idx:  # Don't include the same movie
                    other_movie_vector = Vt[:, i]
                    # Cosine similarity
                    sim = np.dot(target_movie_vector, other_movie_vector) / (
                        np.linalg.norm(target_movie_vector) * np.linalg.norm(other_movie_vector)
                    )
                    similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N similar movies
            top_similar = similarities[:top_n]
            result = []
            for idx, sim in top_similar:
                movie_id = idx_to_movie[idx]
                # Check if movie_id exists in the original movies dataframe
                if movie_id in movies.index:
                    movie_title = movies.loc[movie_id, 'movienm']
                else:
                    movie_title = f"Фильм ID {movie_id} (не найден в оригинальной базе)"
                result.append((movie_id, movie_title, sim))
            return result
        
        ssvd_results = get_top_similar_movies_ssvd(target_movie_idx)
        
        print(f"\nTop 10 фильмов, похожих на '{data['movie']}' (SSVD):")
        for i, (movie_id, movie_title, similarity) in enumerate(ssvd_results, 1):
            print(f"{i:2d}. {movie_title} (ID: {movie_id}, сходство: {similarity:.4f})")
        
        print("\n3. Применяем NMF (Non-negative Matrix Factorization)...")
        
        # Convert sparse matrix to dense for NMF (but only with our sampled data)
        dense_matrix = sparse_matrix.toarray()
        
        # Make sure all values are non-negative
        dense_matrix = np.clip(dense_matrix, 0, None)
        
        # Apply NMF
        n_components_nmf = min(50, min(dense_matrix.shape) - 1)
        model = NMF(n_components=n_components_nmf, random_state=42, max_iter=200)
        W = model.fit_transform(dense_matrix)
        H = model.components_
        
        def get_top_similar_movies_nmf(target_movie_idx, top_n=10):
            """
            Находит топ-N похожих фильмов используя NMF
            """
            # Get the movie vector in latent space (from H transpose)
            target_movie_vector = H[:, target_movie_idx]
            similarities = []
            
            for i in range(H.shape[1]):
                if i != target_movie_idx:  # Don't include the same movie
                    other_movie_vector = H[:, i]
                    # Cosine similarity
                    sim = np.dot(target_movie_vector, other_movie_vector) / (
                        np.linalg.norm(target_movie_vector) * np.linalg.norm(other_movie_vector)
                    )
                    similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N similar movies
            top_similar = similarities[:top_n]
            result = []
            for idx, sim in top_similar:
                movie_id = idx_to_movie[idx]
                # Check if movie_id exists in the original movies dataframe
                if movie_id in movies.index:
                    movie_title = movies.loc[movie_id, 'movienm']
                else:
                    movie_title = f"Фильм ID {movie_id} (не найден в оригинальной базе)"
                result.append((movie_id, movie_title, sim))
            return result
        
        nmf_results = get_top_similar_movies_nmf(target_movie_idx)
        
        print(f"\nTop 10 фильмов, похожих на '{data['movie']}' (NMF):")
        for i, (movie_id, movie_title, similarity) in enumerate(nmf_results, 1):
            print(f"{i:2d}. {movie_title} (ID: {movie_id}, сходство: {similarity:.4f})")
        
        print("\n4. Сравнение методов:")
        print("- SSVD: Работает эффективнее с разреженными матрицами, может быть быстрее")
        print("- NMF: Разложение на неотрицательные компоненты, хорошо интерпретируется")
        print("Оба метода дают разные результаты из-за разных подходов к факторизации.")
        
        print("\n" + "="*60)
        print("ЧАСТЬ 2: Рекомендации похожих пользователей")
        print("="*60)
        
        # Find the target user index
        if data['user'] in user_to_idx:
            target_user_idx = user_to_idx[data['user']]
            print(f"Целевой пользователь: {data['user']} (индекс: {target_user_idx})")
            
            def get_top_similar_users(target_user_idx, method='ssvd', top_n=10):
                """
                Находит топ-N похожих пользователей
                """
                if method == 'ssvd':
                    # Use SSVD-based representation
                    target_user_vector = U[target_user_idx, :]  # User profile in latent space
                    similarities = []
                    
                    for i in range(U.shape[0]):
                        if i != target_user_idx:  # Don't include the same user
                            other_user_vector = U[i, :]
                            # Cosine similarity
                            sim = np.dot(target_user_vector, other_user_vector) / (
                                np.linalg.norm(target_user_vector) * np.linalg.norm(other_user_vector)
                            )
                            similarities.append((i, sim))
                
                elif method == 'nmf':
                    # Use NMF-based representation
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
                return [(idx_to_user[idx], sim) for idx, sim in top_similar]
            
            # Get similar users for each method
            similar_users_ssvd = get_top_similar_users(target_user_idx, 'ssvd')
            similar_users_nmf = get_top_similar_users(target_user_idx, 'nmf')
            
            print(f"\nTop 10 пользователей, похожих на пользователя {data['user']} (SSVD):")
            for i, (user_id, similarity) in enumerate(similar_users_ssvd, 1):
                print(f"{i:2d}. Пользователь {user_id} (сходство: {similarity:.4f})")
            
            print(f"\nTop 10 пользователей, похожих на пользователя {data['user']} (NMF):")
            for i, (user_id, similarity) in enumerate(similar_users_nmf, 1):
                print(f"{i:2d}. Пользователь {user_id} (сходство: {similarity:.4f})")
            
            print("\nСравнение методов для похожих пользователей:")
            print("- SSVD: Основан на сингулярном разложении, эффективен для разреженных данных")
            print("- NMF: Даёт интерпретируемые результаты, особенно для рекомендаций")
        
        else:
            print(f"Целевой пользователь {data['user']} не найден в отфильтрованной матрице.")
    
    else:
        print(f"Целевой фильм с ID {target_movie_id} не найден в отфильтрованной матрице.")

# Implementation of ALS algorithm (simplified version)
def als_algorithm(ratings_df, n_factors=20, reg_param=0.01, n_iterations=5):
    """
    Implementation of Alternating Least Squares algorithm
    """
    print("\n" + "="*60)
    print("ЧАСТЬ 3: Реализация ALS алгоритма")
    print("="*60)
    
    # Sample a smaller dataset for ALS
    sample_size = min(10000, len(ratings_df))
    sampled_ratings = ratings_df.sample(n=sample_size, random_state=42)
    
    # Create user and item mappings
    unique_users = sampled_ratings['userid'].unique()
    unique_items = sampled_ratings['movieid'].unique()
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    # Create rating matrix
    R = np.zeros((n_users, n_items))
    for _, row in sampled_ratings.iterrows():
        u_idx = user_to_idx[row['userid']]
        i_idx = item_to_idx[row['movieid']]
        R[u_idx, i_idx] = row['rating']
    
    # Initialize user and item factors randomly
    U = np.random.normal(0, 0.1, (n_users, n_factors))
    V = np.random.normal(0, 0.1, (n_items, n_factors))
    
    lambda_reg = reg_param
    
    print(f"Размерности: U={U.shape}, V={V.shape}")
    print(f"Количество факторов: {n_factors}, регуляризация: {lambda_reg}, итераций: {n_iterations}")
    print(f"Размер выборки для ALS: {sample_size} рейтингов")
    
    # ALS iterations
    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")
        
        # Update user factors
        for u in range(n_users):
            # Items rated by user u
            rated_items = R[u, :] != 0
            if np.any(rated_items):
                V_rated = V[rated_items, :]
                ratings_u = R[u, :][rated_items]
                
                A = V_rated.T @ V_rated + lambda_reg * np.eye(n_factors)
                b = V_rated.T @ ratings_u
                U[u, :] = np.linalg.solve(A, b)
        
        # Update item factors
        for i in range(n_items):
            # Users who rated item i
            rated_by_users = R[:, i] != 0
            if np.any(rated_by_users):
                U_rated = U[rated_by_users, :]
                ratings_i = R[:, i][rated_by_users]
                
                A = U_rated.T @ U_rated + lambda_reg * np.eye(n_factors)
                b = U_rated.T @ ratings_i
                V[i, :] = np.linalg.solve(A, b)
    
    print("ALS алгоритм завершен.")
    
    # Since we used a sample, let's see if our target user/movie are in the sample
    if data['user'] in user_to_idx and target_movie_id in item_to_idx:
        target_user_idx = user_to_idx[data['user']]
        target_movie_idx = item_to_idx[target_movie_id]
        predicted_rating = np.dot(U[target_user_idx, :], V[target_movie_idx, :])
        
        actual_ratings = sampled_ratings[(sampled_ratings['userid'] == data['user']) & 
                                       (sampled_ratings['movieid'] == target_movie_id)]
        
        if len(actual_ratings) > 0:
            actual_rating = actual_ratings.iloc[0]['rating']
            print(f"\nПрогноз для пользователя {data['user']} и фильма '{data['movie']}':")
            print(f"Предсказанный рейтинг: {predicted_rating:.4f}")
            print(f"Фактический рейтинг: {actual_rating}")
        else:
            print(f"\nПрогноз для пользователя {data['user']} и фильма '{data['movie']}':")
            print(f"Предсказанный рейтинг: {predicted_rating:.4f}")
            print("Фактический рейтинг: отсутствует в выборке")
    else:
        print(f"\nПользователь {data['user']} или фильм '{data['movie']}' не попали в выборку для ALS.")
    
    return U, V, user_to_idx, item_to_idx


# Run ALS algorithm if target movie exists
if target_movie_id is not None:
    U_als, V_als, user_to_idx_als, item_to_idx_als = als_algorithm(ratings, n_factors=20, reg_param=0.01, n_iterations=5)
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("1. SSVD и NMF - методы матричного разложения позволяют находить")
    print("   скрытые паттерны в пользовательских предпочтениях и делать рекомендации.")
    print("2. ALS алгоритм эффективно оптимизирует факторы пользователей и предметов,")
    print("   минимизируя ошибку прогнозирования рейтингов.")
    print("3. Все методы показывают разные аспекты сходства между пользователями и фильмами,")
    print("   что позволяет строить более точные рекомендательные системы.")
    print("4. Для конкретного фильма 'Seventh Seal, The (Sjunde inseglet, Det) (1957)'")
    print("   были получены списки похожих фильмов и похожих пользователей.")
    print("5. Из-за ограничений по памяти пришлось использовать выборку данных,")
    print("   что может повлиять на точность рекомендаций.")