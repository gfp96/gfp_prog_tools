import numpy as np


class Water:
    """Defining water properties"""

    def __init__(self):
        self.rhow = 1000 #kg/m3
        self.Cw =  4.19e-10 # Fredlund 4.5248e-10 #compressibility at 25°C in Pa^-1
        self.Pref = 101325 #Pa
        self.Tref = 298.15 #25°C
        #unknown origin : replaced by data from fredlund
        # self.Cws = np.array(([273.15,5.0879e-10], [278.15, 4.9175e-10], [283.15, 4.7816e-10], [288.15, 4.6739e-10], [293.15, 4.5895e-10], [298.15, 4.5248e-10], [303.15, 4.4771e-10]), dtype = float)
        #data from Dorsey1940 (from Fredlund 2012)
        self.Cws = np.array(([273.15, 4.76e-10], [278.15, 4.59e-10], [283.15, 4.45e-10], [288.15, 4.31e-10], [293.15, 4.19e-10], [298.15, 4.10e-10], [303.15, 4.01e-10]), dtype = float)
        self.minT = 273.15
        self.maxT = 303.15
        self.dyn_visc = 0.89e-3 #dynamic viscosity (Pa.s at 25°C)
        self.kin_visc = 0.884e-6 #kinematic viscosity (m2/s at 25°C)
        
    def Get_vapour_pressure(self, T):
        self.pwv = 0.611*np.exp((17.27*(T-273)/(T+237.3-273)))*1e3 #in Pa # water vapour pressure according to Musy (2005), from Boutonnier (2007)
        return self.pwv

    def Update_dynamic_viscosity(self, T): #T in K from Viswanath & Natavajan (1989)
        self.dyn_visc = 0.02939*np.exp(507.88/(T-149.3))/1e3
        return self.dyn_visc


    def Update_Cw(self, Temp):
        index = np.argmin(np.abs(self.Cws[:, 0]-Temp))
        if Temp>= self.maxT:
            self.Cw = self.Cws[index, 1]
        elif Temp<= self.minT:
            self.Cw = self.Cws[index, 1]
        else:
            if self.Cws[index, 0]<=Temp:
                self.Cw = self.Cws[index, 1]+(self.Cws[index+1, 1]-self.Cws[index, 1])/(self.Cws[index+1, 0]-self.Cws[index, 0])*(Temp-self.Cws[index, 0])
            else:
                self.Cw = self.Cws[index, 1]+(self.Cws[index-1, 1]-self.Cws[index, 1])/(self.Cws[index-1, 0]-self.Cws[index, 0])*(Temp-self.Cws[index, 0])
        return self.Cw