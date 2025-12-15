import numpy as np
import random


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


# Пример использования
if __name__ == "__main__":
    # Создадим искусственные данные для тестирования
    # Формат: (user_id, item_id, rating)
    train_data = np.array([
        [0, 0, 5],
        [0, 1, 4],
        [0, 2, 3],
        [1, 0, 4],
        [1, 1, 5],
        [1, 2, 2],
        [2, 0, 3],
        [2, 1, 2],
        [2, 2, 5],
    ])
    
    test_data = np.array([
        [0, 3, 4],
        [1, 3, 3],
        [2, 3, 4],
    ])
    
    # Создаем и обучаем модель
    model = FunkSVD(n_factors=10, n_epochs=50, lr=0.005, reg=0.02)
    model.fit(train_data, test_data)
    
    # Тестируем предсказания
    print("\nПредсказания для тестовых данных:")
    for sample in test_data:
        user_id, item_id, true_rating = sample
        pred_rating = model.predict_single(int(user_id), int(item_id))
        print(f"User {int(user_id)}, Item {int(item_id)}: True={true_rating}, Predicted={pred_rating:.3f}")