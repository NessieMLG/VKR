import time
import numpy as np
from scipy import linalg
from physics_1d import PhysicsEngine1D
from MPC_1d import LQRController1D

def run_pipeline():
    print("=== ЭТАП 1: Сбор дискретных 1D данных ===")
    engine = PhysicsEngine1D("1-d variant/models/mechanism_1d.xml")
    
    states = []
    next_states = []
    inputs = []
    
    # Собираем данные небольшими сессиями, сбрасывая шар, если он падает
    for session in range(20):
        engine.reset_ball(fixed_offset=np.random.uniform(-0.04, 0.04))
        for _ in range(150):
            z = engine.get_state()
            
            # ИСПРАВЛЕНИЕ: Делаем шум сбора данных экстремально нежным (в пределах +- 0.5 градуса)
            u = np.array([np.random.uniform(-0.01, 0.01)]) 
            
            states.append(z)
            inputs.append(u)

            
            engine.set_motor_position(u[0])
            engine.step()
            
            next_states.append(engine.get_state())
            if abs(z[0]) > 0.075: # Если шар улетел к краю, прерываем сессию
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
    
    Q = np.diag([50.0, 0.5])
    R = np.array([[1000.0]])


    
    controller = LQRController1D(A_disc, B_disc, Q, R)

    print("\n=== ЭТАП 4: Финальный тест системы в реальном времени ===")
    engine.reset_ball(fixed_offset=0.04) # Ставим шар в начальный перекос +4 см
    
    import mujoco.viewer
    t = 0
    control_frequency = 5  # Вызываем LQR каждые 5 шагов физики (частота 200 Гц)
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
                print(f"Время: {t*0.001:.2f}с | Положение шара X: {state[0]*100:.2f} см | Угол мотора U: {np.degrees(u_control):.2f}°")

if __name__ == "__main__":
    run_pipeline()
