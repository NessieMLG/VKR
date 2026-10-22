import time
import numpy as np
from physics import PhysicsEngine
from MPC import MPCController

def run_mpc_simulation():
    engine = PhysicsEngine("models/mechanism.xml")
    engine.reset_ball()
    
    # Load matrices from identification
    try:
        A = np.loadtxt("A_matrix.csv", delimiter=",")
        B = np.loadtxt("B_matrix.csv", delimiter=",")
    except FileNotFoundError:
        print("Error: Identification matrices not found. Run identify_system.py first.")
        return

    # Define cost matrices
    Q = np.diag([100, 1, 100, 1]) # High weight on position error
    R = np.diag([1, 1])           # Weight on control effort
    
    mpc = MPCController(A, B, Q, R)
    
    print("Starting MPC Simulation...")
    
    for t in range(500):
        state = engine.get_state()
        u = mpc.compute_control(state)
        
        engine.set_motor_positions(u[0], u[1])
        engine.step()
        
        if t % 50 == 0:
            print(f"Step {t}: Ball Pos = ({state[0]:.4f}, {state[2]:.4f}) | Control = ({u[0]:.4f}, {u[1]:.4f})")

    print("MPC Simulation complete.")

if __name__ == "__main__":
    run_mpc_simulation()
