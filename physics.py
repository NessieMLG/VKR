class axis:
    angle = 0.0 # Angle of an axis, degrees. Level to the ground is 0.0
    turnspeed = 10.0 # Turnspeed of the axis, degrees per second
    position = 0.0 # Position of the ball relative to the axis center, meters
    dt = 0.001 # Time delta, seconds
    drag = 0.00001 # Drag coefficient of a ball, no dimension
    mass = 0.001 # Mass of the ball, kg
    radius = 0.002 # Radius of the ball, kg

    def turn(self, dir):
        self.angle += self.turnspeed*self.dt*dir
        if self.angle > 90.0:
            self.angle = 90.0
        if self.angle < -90.0:
            self.angle = -90.0
        return self
    
    def update_position(self):

       return self.position 
    
    def
