import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from manipulators import Planar3R
from scipy.signal import savgol_filter

pi = np.pi



initial_conditions = [[np.array([0.0, 0.0 , 0.0]), "[0, 0, 0]"], [np.array([pi/2.0, pi/2.0, pi/2.0]), "[π/2, π/2, π/2]"]]
x_e = np.array([1, 2, pi/2.0])


MAX_ITER = 10000

EPS = 1e-4
for init_q, init_q_string in initial_conditions:
    
    robot = Planar3R(init_q)

    k = robot.compute_direct_kin
    J = robot.compute_jacobian

    for alpha in [1/2, 1/10]:
        objective = []
        raw_error = []
        i = 0
        converge = False
        q = init_q.copy()

        while not converge and i < MAX_ITER:
            kin_map = k(q)
            q = q + alpha * (J(q).T @ (x_e - kin_map))
            e = 0.5 * np.linalg.norm(x_e - kin_map)**2
            converge = e < EPS
            objective.append(e)
            raw_error.append(np.linalg.norm(x_e - kin_map))
            i+=1

        print("+ ---------------- +")
        print(f"Start: {init_q} | step_size: {alpha}")
        print(f"Desired: {x_e}")
        print(f"Actual:  {k(q)}")
        
        plt.yscale('log')
        plt.plot(np.linspace(1,i,i), objective, label = 'MSE(e)', color = 'blue',alpha = 0.8)
        plt.axhline(y = EPS, linestyle='--', color='gray', label='tol = 1e-4')

        if not converge:
            objective_smooth = savgol_filter(objective, window_length=30, polyorder=3)
            plt.plot(np.linspace(1,i,i), objective_smooth, label='MSE(e)(filtered)', color = 'red')

        plt.title(f"Gradient method - starting position: {init_q_string}, α = {alpha}")
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.legend()
        
        plt.grid(True, alpha=0.3)

        plt.show()

