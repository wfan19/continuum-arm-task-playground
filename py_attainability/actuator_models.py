
def bellow(strain, pressure):
    """
        Compute the bellow reaction force when it is at a given strain and pressure
        Fitted model based on bellow actuator data collected on Instron
        by Bill Fan from summer 2023

        strain: scalar or np.array
        pressure: scalar
    """
    p00 =           0
    p10 =       10.47
    p01 =      -1.068
    p20 =      -7.344
    p11 =      0.5504
    p02 =     0.00205
    p30 =       19.39
    p21 =    -0.03461
    p12 =   -0.006294
    p03 =   3.081e-05

    force = (p00 + 
            p10 * strain + p01 * pressure + 
            p20 * strain**2  + p11 * strain * pressure + p02 * pressure**2 + 
            p30 * strain**3 + p21 * strain**2 * pressure + p12 * strain * pressure**2 + p03 * pressure**3)
    force = force;
    return force

class ActuatorModel:
    p_bounds = [0, 100]
    e_bounds = [-0.1, 0.1]

    @classmethod
    def plot(fig):
        raise NotImplementedError

    @staticmethod
    def force(strain, pressure):
        pass
    

class Bellow(ActuatorModel):
    
    p_bounds = [0, 50]
    e_bounds = [-0.167, 0.6680]
    
    @staticmethod
    def force(strain, pressure):
        """
            Compute the bellow reaction force when it is at a given strain and pressure
            Fitted model based on bellow actuator data collected on Instron
            by Bill Fan from summer 2023

            strain: scalar or np.array
            pressure: scalar
        """
        p00 =           0
        p10 =       10.47
        p01 =      -1.068
        p20 =      -7.344
        p11 =      0.5504
        p02 =     0.00205
        p30 =       19.39
        p21 =    -0.03461
        p12 =   -0.006294
        p03 =   3.081e-05

        force = (p00 + 
                p10 * strain + p01 * pressure + 
                p20 * strain**2  + p11 * strain * pressure + p02 * pressure**2 + 
                p30 * strain**3 + p21 * strain**2 * pressure + p12 * strain * pressure**2 + p03 * pressure**3)
        force = force;
        return force

class Muscle:
    p_bounds = [0, 100]
    e_bounds = [-0.5, 0.05]

    @staticmethod
    def force(strain, pressure):
        return -strain*pressure