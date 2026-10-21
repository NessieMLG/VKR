import time
import numpy as np
from physics import PhysicsEngine
from PID import PID

def run_simulation(kp, ki, kd, render=False):
    engine = PhysicsEngine("models/mechanism.xml")
    
    # Инициализируем симуляцию случайным положением шара
    engine.reset_ball()
    
    pid_pitch = PID(kp=kp, ki=ki, kd=kd, dt=0.002, desired=0.0)
    pid_roll = PID(kp=kp, ki=ki, kd=kd, dt=0.002, desired=0.0)
    
    total_error = 0.0
    steps = 500
    
    if render:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(engine.model, engine.data) as viewer:
            while(1):
                if not viewer.is_running(): break
                x_err, y_err = engine.get_ball_relative_position()
                
                err = np.sqrt(x_err**2 + y_err**2)
                total_error += err
                
                pitch_out = pid_pitch.calc_out(x_err)
                roll_out = pid_roll.calc_out(y_err)
                
                engine.set_motor_positions(pitch_out, roll_out)
                engine.step()
                viewer.sync()
                time.sleep(engine.model.opt.timestep)
    else:
        # Внутри цикла фонового расчета функции run_simulation:
        for t in range(steps):
            x_err, y_err = engine.get_ball_relative_position()
        
            # Квадрат расстояния — сильнее наказывает за далекий улет шара
            err = x_err**2 + y_err**2
        
            # Добавляем штраф за скорость шара (чтобы PID учился тормозить шар, а не просто раскачивать)
            # qvelсвободного сустава шара: первые 3 элемента — линейные скорости
            ball_vel = engine.data.qvel[engine.ball_qposadr : engine.ball_qposadr + 2]
            vel_penalty = np.sum(ball_vel**2) * 0.1 # Вес штрафа за скорость
        
            total_error += (err + vel_penalty)
        
            pitch_out = pid_pitch.calc_out(x_err)
            roll_out = pid_roll.calc_out(y_err)
        
            engine.set_motor_positions(pitch_out, roll_out)
            engine.step()
        
            # Жесткий штраф за падение
            if np.sqrt(x_err**2 + y_err**2) > 0.075: 
                total_error += (steps - t) * 10.0 # Даем массивный штраф
                break
        
                
    return total_error

def twiddle(tol=0.05):
    p = [15.0, 0.01, 1.0]
    dp = [10.0, 0.5, 5.0]
    
    print("Оцениваем базовые параметры...")
    best_err = run_simulation(p[0], p[1], p[2], render=False)
    
    it = 0
    while sum(dp) > tol:
        print(f"Итерация {it:2d} | Kp={p[0]:.2f}, Ki={p[1]:.2f}, Kd={p[2]:.2f} | Шаг dp: {sum(dp):.3f} | Best Err: {best_err:.4f}")
        for i in range(len(p)):
            p[i] += dp[i]
            err = run_simulation(p[0], p[1], p[2], render=False)
            
            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] -= 2 * dp[i]
                err = run_simulation(p[0], p[1], p[2], render=False)
                
                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                else:
                    p[i] += dp[i]
                    dp[i] *= 0.9
        it += 1
        if it > 5000: break
        
    return p

def main():
    print("--- Запуск автонастройки PID (Twiddle) ---")
    best_pid = twiddle()
    print(f"\nОптимальные коэффициенты: Kp={best_pid[0]:.4f}, Ki={best_pid[1]:.4f}, Kd={best_pid[2]:.4f}")
    
    print("\nЗапуск демонстрации...")
    run_simulation(best_pid[0], best_pid[1], best_pid[2], render=True)

if __name__ == "__main__":
    main()
