import numpy as np
from scipy import linalg

class LQRController1D:
    def __init__(self, A, B, Q, R):
        """
        A: Дискретная матрица системы (2x2) -> [x, vx]_{k+1} = A*[x, vx]_k + B*u_k
        B: Дискретная матрица входа (2x1)
        Q: Штрафы за состояние (2x2)
        R: Штраф за управление (1x1)
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        # Решаем дискретное уравнение Риккати
        P = linalg.solve_discrete_are(A, B, Q, R)
        
        # Вычисляем матрицу обратной связи K для дискретной системы
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        
        print("1D LQR initialized. Identified Feedback Gain K:")
        print(self.K)

    def compute_control(self, state):
        # Вычисляем управление (получаем массив из одного элемента, например, [0.023])
        u_out = -self.K @ state  
        
        # ИСПРАВЛЕНИЕ: Извлекаем первый элемент [0] и превращаем его в float, 
        # чтобы MuJoCo принял его как одиночное число, а не последовательность
        u_scalar = float(u_out[0])
        
        # Ограничиваем угол в безопасном диапазоне (примерно +- 4.5 градуса)
        return np.clip(u_scalar, -0.150, 0.150)
