import numpy as np
from scipy import linalg

def identify_system():
    # Load data
    data = np.loadtxt("data_combined.csv", delimiter=",") # [x, vx, y, vy, u1, u2]
    targets = np.loadtxt("data_targets.csv", delimiter=",") # [x_next, vx_next, y_next, vy_next]
    
    # We want to find A and B such that z_next = A*z + B*u
    # where z = [x, vx, y, vy]^T and u = [u1, u2]^T
    # Let M = [A | B] be the transition matrix.
    # We solve M * data = targets using Least Squares.
    
    # Data is (N, 6), Targets is (N, 4)
    # We want to find a (4, 6) matrix M such that M @ data.T = targets.T
    # Solve for M such that M @ data.T = targets.T --> M is (4x6)
    # Since lstsq solves Ax=b where x is the unknown, we use A=data.T and b=targets.T
    # But wait, if A is (6xN) and b is (4xN), it won't work because rows must match.
    # We need to solve for M such that Z_next = M @ Z^T. 
    # This means we want M * data.T = targets.T.
    # To use lstsq(A, b) where A is (4xN), we can't easily do it this way.
    # Let's use the property: M = targets @ inv(data). 
    # Or more robustly: solve for M_transpose such that data @ M_transpose = targets
    M_transpose = linalg.lstsq(data, targets)[0]
    M = M_transpose.T
    
    
    A = M[:, :4]
    B = M[:, 4:]
    
    # Convert to continuous time: z_dot = A_cont * z + B_cont * u
    # Since z(k+1) = (I + dt*A_cont)*z(k) + (dt*B_cont)*u(k)
    # We can approximate A_cont = (A_disc - I) / dt
    # and B_cont = B_disc / dt
    
    dt = 0.001
    # Удалить этот блок:
    # A_cont = (A - np.eye(4)) / dt
    # B_cont = B / dt

    # Вместо этого сохранить чистые A и B:
    np.savetxt("A_matrix.csv", A, delimiter=",")
    np.savetxt("B_matrix.csv", B, delimiter=",")    
    
    print("Identified A matrix:")
    print(A)
    print("\nIdentified B matrix:")
    print(B)

if __name__ == "__main__":
    identify_system()
