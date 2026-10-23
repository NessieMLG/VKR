import numpy as np
from scipy import linalg

class LQRController2D:
    def __init__(self, A, B, Q, R):
        """
        A: Дискретная матрица системы (4x4) -> [x, y, vx, vy]_{k+1}
        B: Дискретная матрица входа (4x2)
        Q: Штрафы за состояние (4x4)
        R: Штраф за управление (2x2)
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        # Математика Риккати сама подстроится под новые 2D-размерности матриц!
        P = linalg.solve_discrete_are(A, B, Q, R)
        
        # Вычисляем матрицу обратной связи K (она станет размером 2x4)
        self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        
        print("2D LQR initialized. Identified Feedback Gain K (Shape 2x4):")
        print(self.K)

    def compute_control(self, state):
        """
        state: [x, y, vx, vy] — длина 4
        Возвращает: np.array([u_motor1, u_motor2]) — длина 2
        """
        # В 2D результатом умножения матрицы K (2x4) на вектор состояния (4x1) 
        # автоматически будет вектор управления из 2 элементов (для motor1 и motor2)
        u_out = -self.K @ state  
        
        # Ограничиваем углы обоих моторов в безопасном диапазоне (из вашего XML: ctrlrange="-0.15 0.15")
        u_clipped = np.clip(u_out, -0.150, 0.150)
        
        # Возвращаем массив из двух элементов для передачи в self.data.ctrl[:]
        return u_clipped