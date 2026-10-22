import mujoco
import numpy as np

class PhysicsEngine1D:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.platform_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        
        # Получаем стартовый адрес свободного сустава шара
        self.ball_qposadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")]
        self.ball_qveladr = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")]

    def reset_ball(self, fixed_offset=0.04):
        mujoco.mj_resetData(self.model, self.data)
        
        # Задаем смещение по X (0-й элемент свободного сустава)
        self.data.qpos[self.ball_qposadr] = fixed_offset
        # Обнуляем смещение по Y (1-й элемент), чтобы шар не улетал в бок
        self.data.qpos[self.ball_qposadr + 1] = 0.0
        
        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        """Возвращает строго 1D координату положения и линейную скорость по X"""
        ball_pos = self.data.xpos[self.ball_id]
        platform_pos = self.data.xpos[self.platform_id]
        
        rel_x = ball_pos[0] - platform_pos[0]
        
        # Линейная скорость по X — это самый первый элемент в qvel свободного сустава
        vx = self.data.qvel[self.ball_qveladr]
        
        return np.array([rel_x, vx])


    def set_motor_position(self, pitch_pos):
        # Просто передаем число в управляющий вектор
        self.data.ctrl = pitch_pos

    def step(self):
        mujoco.mj_step(self.model, self.data)
