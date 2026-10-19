class PID:
    P = 0.0
    I = 0.0
    D = 0.0

    Kp = 1.0
    Ki = 0.2
    Kd = 0.001

    dt = 0.001
    prev_error = 0.0

    desired = 0.0
    output = 0.0

    def calc_out(self, input):
        error = self.desired - input

        self.P = self.Kp * error
        self.I = self.I + self.Ki * error * self.dt
        self.D = self.Kd * (error - self.prev_error) / self.dt

        self.output = self.P + self.I + self.D

        self.prev_error = error

        return self.output
    
    def reset(self):
        self.P = 0.0
        self.I = 0.0
        self.D = 0.0
        self.prev_error = 0.0
        self.desired = 0.0
        self.output = 0.0

        return self
