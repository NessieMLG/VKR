import mujoco
import numpy as np

class PhysicsEngine:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Кэшируем ID, чтобы не искать их в цикле на каждом шаге (повышает скорость)
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.platform_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        
        # Находим ID свободного сустава шара для управления его позицией напрямую
        self.ball_qposadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")]

    def reset_ball(self):
        """Сбрасывает систему в фиксированное начальное состояние для тестов"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Спавним шар ВСЕГДА в одной и той же точке (например, 4 см по X и 4 см по Y)
        # Это даст честную оценку для Twiddle
        self.data.qpos[self.ball_qposadr : self.ball_qposadr + 2] = [0.1, 0.1175]
        
        mujoco.mj_forward(self.model, self.data)

    def set_motor_positions(self, pitch_pos, roll_pos):
        # Передаем углы. Работает корректно ТОЛЬКО если актуаторы в XML изменены на <position>!
        self.data.ctrl[0] = pitch_pos
        self.data.ctrl[1] = roll_pos

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_ball_position(self):
        return self.data.xpos[self.ball_id]

    def get_ball_relative_position(self):
        # Возвращаем только X и Y координаты (2D вектор ошибки)
        ball_pos = self.data.xpos[self.ball_id]
        platform_pos = self.data.xpos[self.platform_id]
        return ball_pos[0] - platform_pos[0], ball_pos[1] - platform_pos[1]
