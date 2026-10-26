import time
import numpy as np
from scipy import linalg
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# Импортируем ваши отлаженные 2D компоненты
from physics import PhysicsEngine2D
from MPC import LQRController2D

def circle_target(scale=0.2, mesh=200000, current_step=0, dt=0.0001):
    # Полное время одного круга в секундах: mesh * dt
    # Угловая скорость (рад/с): omega = 2 * pi / T_circle
    omega = (2 * np.pi) / (mesh * dt)
    
    # Текущий угол
    alpha = omega * (current_step * dt)
    
    # Целевое положение
    x_t = scale * np.cos(alpha)
    y_t = scale * np.sin(alpha)
    
    # Целевая скорость (аналитическая производная)
    vx_t = -scale * np.sin(alpha) * omega
    vy_t = scale * np.cos(alpha) * omega
    
    return np.array([x_t, y_t, vx_t, vy_t])


class PID2D:
    def __init__(self, kp=1.0, ki=0.2, kd=0.001, dt=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.i_term = 0.0
        self.prev_error = 0.0

    def calc_out(self, desired, current_value):
        error = desired - current_value
        p_term = self.kp * error
        self.i_term += self.ki * error * self.dt
        self.i_term = np.clip(self.i_term, -0.05, 0.05)
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        return np.clip(p_term + self.i_term + d_term, -0.150, 0.150)

def run_pipeline():
    xml_path = "models/mechanism.xml"
    engine = PhysicsEngine2D(xml_path)

    print("=== ЭТАП 1: Сбор дискретных 2D данных (Импульсное возбуждение) ===")
    states, next_states, inputs = [], [], []
    K_STEPS = 20  # Оставляем макро-шаг контроллера

    for session in range(2000): # Можно увеличить число коротких сессий
        # Сбрасываем строго в центр, чтобы было больше времени на сбор данных
        engine.reset_ball(offset_x=np.random.uniform(-0.02, 0.02), 
                          offset_y=np.random.uniform(-0.02, 0.02))
        
        # Начальные случайные углы платформы (в радианах, примерно до 7 градусов)
        u_x = np.random.uniform(-0.12, 0.12)
        u_y = np.random.uniform(-0.12, 0.12)
        u = np.array([u_x, u_y])
        
        # Перед циклом step инициализируем счетчик
        steps_to_hold = 0
        u = np.array([0.0, 0.0])

        for step in range(400):
            z_start = engine.get_state()
            
            # Если время удержания предыдущего шага истекло — генерируем новый импульс
            if steps_to_hold <= 0:
                # Выбираем случайную длительность следующей ступеньки (от 5 до 25 макро-шагов)
                steps_to_hold = np.random.randint(5, 25)
                
                # Генерируем случайные импульсы наклона
                u_x = np.random.uniform(-0.12, 0.12)
                u_y = np.random.uniform(-0.12, 0.12)
                
                # Ваша отличная защита: "возврат к центру"
                if z_start[0] > 0.05:  u_x = np.random.uniform(-0.12, 0.0)
                if z_start[0] < -0.05: u_x = np.random.uniform(0.0, 0.12)
                if z_start[1] > 0.05:  u_y = np.random.uniform(-0.12, 0.0)
                if z_start[1] < -0.05: u_y = np.random.uniform(0.0, 0.12)
                
                u = np.array([u_x, u_y])
            
            # Уменьшаем счетчик удержания на каждом шаге цикла
            steps_to_hold -= 1
            
            # Подаем управление на K_STEPS шагов симулятора
            for _ in range(K_STEPS):
                engine.set_motors_positions(u)
                engine.step()
            
            z_end = engine.get_state()
            
            # Сохраняем только если шар гарантированно не врезался в борт
            if abs(z_end[0]) < 0.45 and abs(z_end[1]) < 0.45:
                states.append(z_start)
                inputs.append(u)
                next_states.append(z_end)
            else:
                break # Если долетел до края, мгновенно перезапускаем сессию

    X_data = np.hstack([np.array(states), np.array(inputs)])
    Y_data = np.array(next_states)
    print(f"Собрано успешно {len(X_data)} семплов.")

    print("\n=== ЭТАП 2: Линейная идентификация Least Squares (2D) ===")
    M_T = linalg.lstsq(X_data, Y_data)[0]
    M = M_T.T
    A_disc = M[:, :4]
    B_disc = M[:, 4:]
    print("Идентифицированная 4х4 матрица A:\n", A_disc)
    print("Идентифицированная 4х2 матрица B:\n", B_disc)

    print("\n=== ЭТАП 3: Инициализация класса 2D LQR ===")
    Q = np.diag([5000.0, 5000.0, 100.0, 100.0])  
    R = np.diag([0.0001, 0.0001]) 
    
    # Весь расчет уравнения Риккати ушел внутрь вашего класса!
    controller = LQRController2D(A_disc, B_disc, Q, R)

    print("\n=== ЭТАП 4: Тест 2D LQR контроллера ===")
    engine.reset_ball(offset_x=0.2, offset_y=0.0)
    control_frequency = K_STEPS
    t = 0
    u_control = np.array([0.0, 0.0])

    with mujoco.viewer.launch_passive(engine.model, engine.data) as viewer:
        while True: # Тестируем LQR первые 4 секунды
            if not viewer.is_running(): break
            state = engine.get_state()

            target = state - circle_target(current_step=t, dt=engine.model.opt.timestep)
            
            if t % control_frequency == 0:
                # Метод класса теперь сам умножает матрицы и делает np.clip() для двух осей
                u_control = controller.compute_control(state) 
                
            engine.set_motors_positions(u_control)
            engine.step()
            viewer.sync()
            t += 1
            if t % 5000 == 0:
                print(f"[LQR] X: {state[0]*100:.1f} см | Y: {state[1]*100:.1f} см | U1: {np.degrees(u_control[0]):.1f}° | U2: {np.degrees(u_control[1]):.1f}°")

    print("\n=== ЭТАП 5: Тест 2D LQR контроллера с траекторией ===")
    engine.reset_ball(offset_x=0.2, offset_y=0.0)
    control_frequency = K_STEPS
    t = 0
    u_control = np.array([0.0, 0.0])

    with mujoco.viewer.launch_passive(engine.model, engine.data) as viewer:
        while True: # Тестируем LQR первые 4 секунды
            if not viewer.is_running(): break
            state = engine.get_state()
    
            target = state - circle_target(current_step=t, dt=engine.model.opt.timestep)
                
            if t % control_frequency == 0:
                # Метод класса теперь сам умножает матрицы и делает np.clip() для двух осей
                u_control = controller.compute_control(target) 
                    
            engine.set_motors_positions(u_control)
            engine.step()
            viewer.sync()
            t += 1
            if t % 5000 == 0:
                print(f"[LQR] X: {state[0]*100:.1f} см | Y: {state[1]*100:.1f} см | U1: {np.degrees(u_control[0]):.1f}° | U2: {np.degrees(u_control[1]):.1f}°")

    print("\n=== ЭТАП 5: Тест независимых 2D PID контроллеров ===")
    engine.reset_ball(offset_x=0.08, offset_y=-0.08)
    dt_ctrl = engine.model.opt.timestep * control_frequency
    
    pid_x = PID2D(kp=-15.0, ki=-2.0, kd=-1.2, dt=dt_ctrl)
    pid_y = PID2D(kp=-15.0, ki=-2.0, kd=-1.2, dt=dt_ctrl)
    
    t = 0
    with mujoco.viewer.launch_passive(engine.model, engine.data) as viewer:
        while True:
            if not viewer.is_running(): break
            state = engine.get_state()
            
            if t % control_frequency == 0:
                u_x = pid_x.calc_out(desired=0.0, current_value=state[0])
                u_y = pid_y.calc_out(desired=0.0, current_value=state[1])
                u_control = np.array([u_x, u_y])
                
            engine.set_motors_positions(u_control)
            engine.step()
            viewer.sync()
            t += 1
            if t % 5000 == 0:
                print(f"[PID] X: {state[0]*100:.1f} см | Y: {state[1]*100:.1f} см | U1: {np.degrees(u_control[0]):.1f}° | U2: {np.degrees(u_control[1]):.1f}°")

if __name__ == "__main__":
    run_pipeline()
