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


# Run ALS algorithm
if target_movie_id is not None:
    U_als, V_als, user_to_idx, item_to_idx, predict_fn = als_algorithm(ratings, n_factors=50, reg_param=0.01, n_iterations=10)
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("1. SVD, SSVD и NMF - все три метода матричного разложения позволяют находить")
    print("   скрытые паттерны в пользовательских предпочтениях и делать рекомендации.")
    print("2. ALS алгоритм эффективно оптимизирует факторы пользователей и предметов,")
    print("   минимизируя ошибку прогнозирования рейтингов.")
    print("3. Все методы показывают разные аспекты сходства между пользователями и фильмами,")
    print("   что позволяет строить более точные рекомендательные системы.")
    print("4. Для конкретного фильма 'Seventh Seal, The (Sjunde inseglet, Det) (1957)'")
    print("   были получены списки похожих фильмов и похожих пользователей.")