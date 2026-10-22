import numpy as np
from physics import PhysicsEngine
import csv

def collect_data(steps=50000, dt=0.001):
    engine = PhysicsEngine("models/mechanism.xml")
    engine.reset_ball()
    
    states = []
    next_states = []
    inputs = []
    
    for _ in range(steps):
        z = engine.get_state()
        u = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
        
        states.append(z)
        inputs.append(u)
        
        engine.set_motor_positions(u[0], u[1])
        engine.step()
        next_states.append(engine.get_state())
        # В цикле сбора данных:
        if np.linalg.norm(np.array(z)[[0, 2]]) > 0.15:
            engine.reset_ball()
    
    return np.array(states), np.array(next_states), np.array(inputs)

if __name__ == "__main__":
    print("Collecting data...")
    s, ns, i = collect_data()
    # Save as a single matrix for easier handling: [z_k, u_k] -> z_{k+1}
    data = np.hstack([s, i])
    targets = ns
    np.savetxt("data_combined.csv", data, delimiter=",")
    np.savetxt("data_targets.csv", targets, delimiter=",")
    print(f"Saved {len(s)} samples.")
