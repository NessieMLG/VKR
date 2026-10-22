import numpy as np
from scipy import linalg

class MPCController:  # Оставляем имя класса, чтобы не ломать ваш main.py
    def __init__(self, A, B, Q, R):
        """
        A: Дискретная матрица системы (4x4) -> z_{k+1} = A*z_k + B*u_k
        B: Дискретная матрица входа (4x2)
        Q: Матрица штрафов за состояние (4x4)
        R: Матрица штрафов за управление (2x2)
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        print("Решаем дискретное уравнение Риккати (DARE)...")
        # 1. ИСПРАВЛЕНИЕ: Используем дискретное уравнение Риккати для MuJoCo шагов
        P = linalg.solve_discrete_are(A, B, Q, R)
        
        # 2. ИСПРАВЛЕНИЕ: Формула K для дискретного LQR принципиально другая!
        # K = (R + B^T * P * B)^-1 * B^T * P * A
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        
        print("Матрица обратной связи K успешно вычислена:")
        print(self.K)

    # Измените в MPC.py метод compute_control:
    def compute_control(self, state):
        # Убираем минус, меняем знак обратной связи на противоположный
        return self.K @ state


def main():
    # Загружаем СТРОГО ДИСКРЕТНЫЕ матрицы (без деления на dt в identify_system.py!)
    try:
        A = np.loadtxt("A_matrix.csv", delimiter=",")
        B = np.loadtxt("B_matrix.csv", delimiter=",")
    except FileNotFoundError:
        print("Ошибка: Сначала соберите данные и запустите идентификацию.")
        return
    
    Q = np.diag([500.0, 10.0, 500.0, 10.0]) # Сильно штрафуем отклонение x и y шара
    R = np.diag([0.1, 0.1])                # Даем моторам больше свободы (меньше штраф за u)
    
    mpc = MPCController(A, B, Q, R)
    print("Дискретный LQR-контроллер готов к работе.")

if __name__ == "__main__":
    main()
