import numpy as np
import matplotlib.pyplot as plt
from manipulators import Planar3R

pi = np.pi



initial_conditions = [[np.array([0.0, 0.0 , 0.0]), "[0, 0, 0]"], [np.array([pi/2.0, pi/2.0, pi/2.0]), "[π/2, π/2, π/2]"]]
x_e = np.array([1, 2, pi/2.0])



EPS = 1e-2
for init_q, init_q_string in initial_conditions:
    
    robot = Planar3R(init_q)

    k = robot.compute_direct_kin
    J = robot.compute_jacobian

    
    objective = []
    i = 0
    converge = False
    q = init_q.copy()

    while not converge and i < 100000:
        kin_map = k(q)
        q = q + (np.linalg.pinv(J(q)) @ (x_e - kin_map))
        e = 0.5 * np.linalg.norm(x_e - kin_map)**2
        converge = e < EPS
        objective.append(e)
        i+=1
            
    print("+ ---------------- +")
    print(f"Start: {init_q}")
    print(f"Desired: {x_e}")
    print(f"Actual:  {k(q)}")
    plt.plot(np.linspace(1,i,i), objective)
    plt.title(f"start: {init_q_string}")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()