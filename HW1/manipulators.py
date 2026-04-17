import numpy as np

# useful trigonometric aliases
c = np.cos
s = np.sin
pi = np.pi

class Planar3R():
    def __init__(self, joints = None):

        self.joints = joints
        self.EE = None

    def get_joints_state(self):
        return self.joints
    
    def get_EE_state(self):
        return self.EE
    
    def compute_direct_kin(self, joints):
        if joints is not None:
            self.joints = joints

        theta_1, theta_2, theta_3 = self.joints

        self.EE = np.array([c(theta_1) + c(theta_1 + theta_2) + c(theta_1 + theta_2 + theta_3),
                         s(theta_1) + s(theta_1 + theta_2) + s(theta_1 + theta_2 + theta_3),
                         theta_1 + theta_2 + theta_3],dtype=float)
        return self.EE

    def compute_jacobian(self, joints):

        theta_1, theta_2, theta_3 = self.joints

        
        return np.array([[-s(theta_1)-s(theta_1 + theta_2)-s(theta_1 + theta_2 + theta_3), -s(theta_1 + theta_2)-s(theta_1 + theta_2 + theta_3), -s(theta_1 + theta_2 + theta_3)],
                          [c(theta_1)+c(theta_1 + theta_2)+c(theta_1 + theta_2 + theta_3), c(theta_1 + theta_2)+c(theta_1 + theta_2 + theta_3), c(theta_1 + theta_2 + theta_3)], 
                          [1,1,1]], dtype=float)
        