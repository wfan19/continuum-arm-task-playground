import numpy as np

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

class Muscle(ActuatorModel):
    p_bounds = [0, 100]
    e_bounds = [-0.5, 0.05]

    @staticmethod
    def force(strain, pressure):

        e_crit = -0.303225592;
        e_lim = -0.274484489;
        
        kIa = 3.695687763
        kIb = [0.637225654, 0.453137516]
        kIIa = [1.038555666, 0.511454549]
        kIIb = [1.963398204, -3.169979385, 4.231857967]
        kIIIa = [265.2968218, 320.5906703, 126.11975, 21.92761236]
        kIIIb = [89998.04265, 836.5810641, 62.59606786]
        kIV = [34610.2847, 5991.799518, -1032.874471, -395.9792747]

        def freeContract_IV(strain):
            return np.polyval(kIV + [0], strain)

        def passiveExt_RIIIb(strain):
            return np.polyval(kIIIb + [0], strain)

        def pressComp_Ia(strain, pressure):
            Pz = freeContract_IV(e_lim)
            
            compForce_limit = np.polyval(kIIIa + [0],e_lim)
            critPressure = (compForce_limit/(kIb[0]*(kIb[1]+e_lim)**2))+Pz
            
            compForce =  np.polyval(kIIIa + [0],strain)
            
            k_bc = (kIb[0]*(kIb[1]+e_lim)**2)/((-e_crit+e_lim)**2)

            
            if pressure<=critPressure:
                f = compForce;
            else:
                if -e_crit+strain>=0:
                    f = compForce + k_bc*((-e_crit+strain)**2)*(pressure-critPressure);
                else:
                    f = compForce - kIa*((-e_crit+strain)**2)*(pressure-critPressure);
            return f

        def pressComp_Ib(strain, pressure, zfp):
            compForce = np.polyval(kIIIa + [0],strain);
            critPressure = (compForce/(kIb[0]*(kIb[1]+strain)**2))+zfp;
            
            if pressure<=critPressure:
                f = compForce;
            else:
                f = compForce + kIb[0]*((kIb[1]+strain)**2)*(pressure-critPressure);
            return f

        def actuation_RIIa(strain,pressure, zfp):
            return kIIa[0]*((kIIa[1]+strain)**2)*(pressure-zfp);

        def pressureExt_RIIb(strain,pressure):
            k_bc = np.sqrt(kIIa[0]*(kIIa[1]**2)/kIIb[0]);
            pE = np.polyval(kIIIb+[0],strain);
            f = pE + kIIb[0]*((k_bc+kIIb[1]*strain)**2)*(pressure**(1+kIIb[2]*strain)); 
            return f

        def passiveComp_RIIIa(strain):
            return np.polyval(kIIIa+[0],strain);

        def compute_scalar_force(strain, pressure):
            if pressure == 0:
                if strain >= 0: # Passive extension
                    force = passiveExt_RIIIb(strain)
                else:           # Passive compression
                    force = passiveComp_RIIIa(strain)
            else:
                if strain >= 0: # Pressurized extension
                    force = pressureExt_RIIb(strain, pressure)
                else:           # Active region or pressurized compression
                    zfp = freeContract_IV(strain)

                    if pressure>=zfp:   # Active region
                        force = actuation_RIIa(strain, pressure, zfp)
                    else:
                        if strain >= e_lim: # Pressurized comp 1b
                            force = pressComp_Ib(strain, pressure, zfp)
                        else:
                            force = pressComp_Ia(strain, pressure)
            return force

        if hasattr(strain, "__len__"):
            forces_out = np.zeros(strain.shape)
            for i, strain_i in enumerate(strain):
                forces_out[i] = -compute_scalar_force(strain_i, pressure)
            return forces_out
        else:
            return -compute_scalar_force