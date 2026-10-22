import time
import numpy as np
from physics import PhysicsEngine
from PID import PID
from MPC import MPCController

def run_mpc_simulation(render=False):
    engine = PhysicsEngine("models/mechanism.xml")
    
    # Load matrices for MPC
    try:
        A = np.loadtxt("A_matrix.csv", delimiter=",")
        B = np.loadtxt("B_matrix.csv", delimiter=",")
    except FileNotFoundError:
        print("Error: Identification matrices not found. Run identify_system.py first.")
        return

    # ИСПРАВЛЕНИЕ 1: Кардинально увеличиваем R для плавности и уменьшаем агрессивность Q
    Q = np.diag([100.0, 1.0, 100.0, 1.0]) # Умеренный штраф за позицию и скорость
    R = np.diag([50.0, 50.0])             # Высокий штраф на моторы — убирает дергание
    mpc = MPCController(A, B, Q, R)

    engine.reset_ball()
    
    total_error = 0.0
    steps = 500
    control_frequency = 10  # Обновляем управление раз в 10 шагов физики (100 Гц)
    u = np.array([0.0, 0.0]) # Текущий буфер команд для моторов
    
    if render:
        import mujoco.viewer
        t = 0
        with mujoco.viewer.launch_passive(engine.model, engine.data) as viewer:
            while True:
                if not viewer.is_running(): break
                state = engine.get_state()
                x_err, y_err = state[0], state[2]
                
                err = np.sqrt(x_err**2 + y_err**2)
                total_error += err
                
                # ИСПРАВЛЕНИЕ 2: Считаем управление LQR только на каждом 10-м шаге
                if t % control_frequency == 0:
                    u = mpc.compute_control(state)
                
                # Применяем зафиксированное управление u на каждом шаге физики
                engine.set_motor_positions(u[0], u[1])
                engine.step()
                viewer.sync()
                time.sleep(engine.model.opt.timestep)
                t += 1
    else:
        for t in range(steps):
            state = engine.get_state()
            x_err, y_err = state[0], state[2]
            
            # Penalty for distance and velocity
            ball_vel = engine.data.qvel[engine.ball_qposadr : engine.ball_qposadr + 2]
            err = (x_err**2 + y_err**2) + np.sum(ball_vel**2) * 0.1
            
            total_error += err
            
            # ИСПРАВЛЕНИЕ 3: Дублируем логику редкого вызова управления для фонового режима
            if t % control_frequency == 0:
                u = mpc.compute_control(state)
                
            engine.set_motor_positions(u[0], u[1])
            engine.step()
            
            # Penalty for falling off (размер платформы 0.15, предел от центра 0.075)
            if np.sqrt(x_err**2 + y_err**2) > 0.075:
                total_error += (steps - t) * 10.0
                break
        
    print(f"MPC Simulation complete. Total Error: {total_error:.4f}")

def main():
    # Run MPC with rendering
    run_mpc_simulation(render=True)

if __name__ == "__main__":
    main()
