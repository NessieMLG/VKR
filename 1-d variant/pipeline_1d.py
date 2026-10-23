import time
import numpy as np
from scipy import linalg
from physics_1d import PhysicsEngine1D
from MPC_1d import LQRController1D

class PID:
    def __init__(self, kp=1.0, ki=0.2, kd=0.001, dt=0.001, desired=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.desired = desired

        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.prev_error = 0.0
        self.output = 0.0

    def calc_out(self, current_value):
        error = self.desired - current_value

        self.p_term = self.kp * error
        self.i_term += self.ki * error * self.dt
        # Optional: Add integral windup protection here if needed
        
        self.d_term = self.kd * (error - self.prev_error) / self.dt

        self.output = self.p_term + self.i_term + self.d_term

        self.prev_error = error

        return np.clip(self.output, -0.150, 0.150)
    
    def reset(self):
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.prev_error = 0.0
        self.desired = 0.0
        self.output = 0.0

        return self

def run_pipeline():
    print("=== ЭТАП 1: Сбор дискретных 1D данных ===")
    engine = PhysicsEngine1D("1-d variant/models/mechanism_1d.xml")

    reg = PID(kp=-309156.1999, ki=-4193908.331, kd=-5697.4037, dt=engine.model.opt.timestep/200)
    
    states = []
    next_states = []
    inputs = []
    
    # ФИЗИЧЕСКИЙ ШАГ ОСТАЕТСЯ 0.0001, но фиксировать данные будем раз в 20 шагов
    K_STEPS = 20

    for session in range(200):
        engine.reset_ball(fixed_offset=np.random.uniform(-0.1, 0.1))
        freq = np.random.uniform(1.0, 3.0) 
        
        for step in range(1000):
            # 1. Запоминаем НАЧАЛЬНОЕ состояние макро-шага
            z_start = engine.get_state()
            
            # Амплитуда управления в радианах
            u_val = 0.15 * np.sin(step * 0.05 * 2 * np.pi * freq)
            u = np.array([u_val]) 
            
            # 2. Крутим физику MuJoCo 200 раз на микро-шаге 0.0001с
            for _ in range(K_STEPS):
                engine.set_motor_position(u[0])
                engine.step()
            
            # 3. Запоминаем КОНЕЧНОЕ состояние макро-шага
            z_end = engine.get_state()
            
            # Сохраняем в датасет связь: z_start + u -> приведи к z_end
            states.append(z_start)
            inputs.append(u)
            next_states.append(z_end)
            
            if abs(z_end[0]) > 0.25: 
                break

    X_data = np.hstack([np.array(states), np.array(inputs)])
    Y_data = np.array(next_states)
    print(f"Собрано успешно {len(X_data)} семплов.")

    print("\n=== ЭТАП 2: Линейная идентификация Least Squares ===")
    # Решаем систему: X_data @ M_T = Y_data  => M_T — матрица переходов транпонированная
    M_T = linalg.lstsq(X_data, Y_data)[0]
    M = M_T.T
    
    A_disc = M[:, :2] # Размерность 2х2
    B_disc = M[:, 2:] # Размерность 2х1
    print("Идентифицированная дискретная матрица A:\n", A_disc)
    print("Идентифицированная дискретная матрица B:\n", B_disc)

    print("\n=== ЭТАП 3: Настройка весов LQR и инициализация ===")
    
    Q = np.diag([500.0, 10.0])  
    R = np.array([[1.0]])


    
    controller = LQRController1D(A_disc, B_disc, Q, R)

    print("\n=== ЭТАП 4: Финальный тест системы в реальном времени ===")
    engine.reset_ball(fixed_offset=0.1) # Ставим шар в начальный перекос +10 см
    
    import mujoco.viewer
    t = 0
    control_frequency = 20  # Вызываем LQR каждые 20 шагов физики
    u_control = 0.0
    
    with mujoco.viewer.launch_passive(engine.model, engine.data) as viewer:
        while True:
            if not viewer.is_running(): break
            
            state = engine.get_state()
            
            if t % control_frequency == 0:
                u_control = controller.compute_control(state)
                
            engine.set_motor_position(u_control)
            engine.step()
            
            viewer.sync()
            time.sleep(engine.model.opt.timestep)
            t += 1
            
            if t % 200 == 0:
                print(f"Время: {t*engine.model.opt.timestep:.2f}с | Положение шара X: {state[0]*100:.2f} см | Угол мотора U: {np.degrees(u_control):.2f}°")

    t = 0

    with mujoco.viewer.launch_passive(engine.model, engine.data) as viewer:
            while True:
                if not viewer.is_running(): break
                
                state = engine.get_state()

                if t % control_frequency == 0:
                    u_control = reg.calc_out(state[0])
                    
                engine.set_motor_position(u_control)
                engine.step()
                
                viewer.sync()
                time.sleep(engine.model.opt.timestep)
                t += 1
                
                if t % 200 == 0:
                    print(f"Время: {t*engine.model.opt.timestep:.2f}с | Положение шара X: {state[0]*100:.2f} см | Угол мотора U: {np.degrees(u_control):.2f}°")
    

if __name__ == "__main__":
    run_pipeline()
