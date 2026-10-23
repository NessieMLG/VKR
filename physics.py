import mujoco
import numpy as np

class PhysicsEngine2D:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.platform_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        
        # Получаем стартовые адреса в qpos и qvel для свободного сустава шара
        self.ball_qposadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")]
        self.ball_qveladr = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")]

    def reset_ball(self, offset_x=0.04, offset_y=0.04):
        mujoco.mj_resetData(self.model, self.data)
        
        # Задаем смещение по X (0-й элемент freejoint) и по Y (1-й элемент freejoint)
        self.data.qpos[self.ball_qposadr] = offset_x
        self.data.qpos[self.ball_qposadr + 1] = offset_y
        
        # Высоту Z (2-й элемент) оставляем по умолчанию (0.4), кватернионы не трогаем
        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        """Возвращает 2D состояние: [относительный_x, относительный_y, vx, vy]"""
        ball_pos = self.data.xpos[self.ball_id]
        platform_pos = self.data.xpos[self.platform_id]
        
        # Считаем относительные координаты по обеим осям плоскости
        rel_x = ball_pos[0] - platform_pos[0]
        rel_y = ball_pos[1] - platform_pos[1]
        
        # Линейные скорости по X и Y — это первые два элемента в qvel свободного сустава
        vx = self.data.qvel[self.ball_qveladr]
        vy = self.data.qvel[self.ball_qveladr + 1]
        
        return np.array([rel_x, rel_y, vx, vy])

    def set_motors_positions(self, u_array):
        """
        Передаем вектор из двух управляющих сигналов:
        u_array[0] -> motor1 (управление по оси X)
        u_array[1] -> motor2 (управление по оси Y)
        """
        # MuJoCo автоматически сопоставит элементы массива с актуаторами в порядке их объявления в XML
        self.data.ctrl[:] = u_array

    def step(self):
        mujoco.mj_step(self.model, self.data)
