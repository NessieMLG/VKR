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

        return self.output
    
    def reset(self):
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.prev_error = 0.0
        self.desired = 0.0
        self.output = 0.0

        return self
