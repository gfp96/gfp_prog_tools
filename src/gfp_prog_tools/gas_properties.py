# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 08:51:32 2021

@author: g.flood-page
"""
# g classes
import seaborn as sns
import numpy as np


class Gas:
    """Defining gas properties"""

    def __init__(g, name):
        g.name = name

    Hcp = 0.0  # Henry solubility - aqueous phase concentration
    Mg = 0.0  # Molecular weight
    Df = 0.0  # Diffusion coefficient water
    # Pg = 0.0 #absolute pressure [Pa]
    colour = "black"
    R = 8.31446261815324  # Ideal g constant [J.K-1.mol-1]
    rho = 0.0  # g density
    Tref = 298.15 # reference temperature [K]
    DHR = 0.0 # Enthalpy of dissolution/R = dlnH/d(1/T) according to Van't Hoff equation - values from Sander 2015
    heat_cap = 1.0 # Heat capacity ratio : https://en.wikipedia.org/wiki/Heat_capacity_ratio -
    # Values given at 20°C, little variation with T - doubt on N2 value

    def Reset(g):  # reset the instance
        g.Hcp = 0.0  # Henry solubility - aqueous phase concentration
        g.Mg = 1.0  # Molecular weight
        g.Df = 0.0  # Diffusion coefficient water
        g.colour = "black"
        g.R = 8.31446261815324  # Ideal g constant [J.K-1.mol-1]
        g.rho = 0.0  # g density
        g.DHR = 0.0 
        g.heat_cap = 1.0

    def Getprops(g):
        """Get gas properties"""
        possible_gas = {"CO2", "N2", "O2", "Air", "Air_D&N_lower", "Air_D&N_upper", "Methane"}
        colours = sns.color_palette("husl", len(possible_gas), as_cmap=False)  # set color scheme
        #ges = {"CO2":"blue", "N2":"green", "O2":"red", "Air_D&N_lower":"grey"}
        assert g.name in possible_gas, "g type not recognised"
        if g.name == "CO2":
            g.Hcp = 3.30e-4  # Sander (2015) mol/(m3Pa)
            g.Mg = 44.009e-3  # Molecular weight  https://en.wikipedia.org/wiki/Carbon_dioxide
            g.Df = 1.92e-9  # at 25°C, from Cussler, 2009
            g.DHR = 2400 #Sander (2015)
            g.heat_cap = 1.3 #https://en.wikipedia.org/wiki/Heat_capacity_ratio
            g.colour = colours[0]
            g.lstyle = [1, 0]  # "solid"
        elif g.name == "N2":
            g.Hcp = 6.40E-06  # Sander(2015)
            g.Mg = 28.0134e-3  # https://fr.wikipedia.org/wiki/Diazote
            g.Df = 1.88e-9  # at 25°C, from Cussler, 2009
            g.DHR = 1300 #Sander (2015)
            g.heat_cap = 1.039 #https://www.ohio.edu/mechanical/thermo/property_tables/gas/idealGas.html
            g.colour = colours[1]
            g.lstyle = [2, 1, 1, 1]  # 2pt dash 1pt space 1pt dash 1pt space
        elif g.name == "O2":
            g.Hcp = 1.30E-05  # Sander(2015)
            g.Mg = 31.9988e-3  # https://fr.wikipedia.org/wiki/Dioxygène
            g.Df = 2.1e-9  # at 25°C, from Cussler, 2009
            g.DHR = 1500 #Sander (2015)
            g.heat_cap = 1.4 #https://en.wikipedia.org/wiki/Heat_capacity_ratio
            g.colour = colours[2]
            g.lstyle = [3, 3, 1, 3]  # 1 pt dash 3 pt space 1 pt dash 3 pt space
        elif g.name == "Air_D&N_lower":
            g.Hcp = 8.15E-06  # Duncan & Needham (2004) : = 0.744/1atm
            g.Mg = 29.0e-3  # Molecular mass : Duncan & Needham (2004)
            g.Df = 2.0e-9  # Diffusion coefficient water  Duncan & Needham (2004)
            g.colour = colours[3]
            g.lstyle = [1, 1]  # 1pt dash 1pt space
        elif g.name == "Air_D&N_upper":
            g.Hcp = 7.34E-06  # Duncan & Needham (2004) : = 0.826/1atm
            g.Mg = 29.0e-3  # Molecular mass : Duncan & Needham (2004)
            g.Df = 1.75e-9  # Diffusion coefficient water  Duncan & Needham (2004)
            g.colour = colours[4]
            g.lstyle = [5, 5]  # 5pt dash 5 pt space
        elif g.name == "Air":  # avg of D&N upper and lower
            N2 = Gas("N2")
            N2.Getprops()
            O2 = Gas("O2")
            O2.Getprops()
            g.Hcp = np.average([N2.Hcp, O2.Hcp], weights = [80, 20]) #Sander 2015 + arithmetic avg
            g.Mg = 29.0e-3  # Molecular mass : Duncan & Needham (2004)
            g.Df = round(((1.75e-9+2.0e-9)/2)*1e9, 3)/1e9  # Diffusion coefficient water  Duncan & Needham (2004)
            g.heat_cap = 1.4 #https://en.wikipedia.org/wiki/Heat_capacity_ratio
            g.colour = colours[5]
            g.lstyle = [3, 2]  # 3 pt dash 2 pt space
        elif g.name == "Air_D&N_avg":  # avg of D&N upper and lower
            g.Hcp = round(((7.34E-06+8.15e-6)/2)*1e6, 3)/1e6  # Duncan & Needham (2004)
            g.Mg = 29.0e-3  # Molecular mass : Duncan & Needham (2004)
            g.Df = round(((1.75e-9+2.0e-9)/2)*1e9, 3)/1e9  # Diffusion coefficient water  Duncan & Needham (2004)
            g.heat_cap = 1.4 #https://en.wikipedia.org/wiki/Heat_capacity_ratio
            g.colour = colours[5]
            g.lstyle = [10, 2]  # 10 pt dash 2 pt space
        elif g.name == "Methane":  #
            g.Hcp = 1.40E-05  # Sander (2015)
            g.Mg = 16.043e-3  # Molecular mass : https://en.wikipedia.org/wiki/Methane
            g.Df = 1.49e-9  # at 25°C, from Cussler, 2009
            g.DHR = 1600 #Sander (2015)
            g.heat_cap = 1.320 #https://en.wikipedia.org/wiki/Heat_capacity_ratio
            g.colour = colours[6]
            g.lstyle = [1, 1, 4, 1]  # 1 pt dash 1 pt space

    def Update_Hcp(g, Temp):
        if g.name in {"Methane", "N2", "O2", "CO2"}:
            g.Hcp = g.Hcp*np.exp(g.DHR*(1/Temp-1/g.Tref)) #DHR = d(ln(Hcp))/d(1/T) : assumed constant and tabulated in Sander (2015). Valid for "limited" temperature range
        elif g.name=="Air":
            N2 = Gas("N2")
            N2.Getprops()
            N2.Update_Hcp(Temp)
            O2 = Gas("O2")
            O2.Getprops()
            O2.Update_Hcp(Temp)
            g.Hcp = np.average([N2.Hcp, O2.Hcp], weights = [80, 20], )
        else:
            print("Variation of Hcp constant with temperature cannot be calculated fo D&N values")
            pass #does not work for D&N air values

    def Getrho(g, Temp, Pg):  # Temp in Kelvin, Pg  : absolute pressure in Pa
        """Get the mass density of the gas at pressure Pg"""
        g.rho = g.Mg*Pg/(g.R*Temp)
        return g.rho

    def GetKhpx(g, rhow):  # rho in kg/m3
        """Get Henry volatility coefficient Khpx"""
        Mw = 18.01528e-3
        g.Khpx = 1/(Mw/rhow*g.Hcp)
        return g.Khpx

    def Geth(g, rhow, T):
        """Get solubility coefficient h"""
        g.h = g.R*T*g.Hcp
        return g.h

    def Get_sbm(g, ts, wetting_angle, rbm):
        """Returns the capillary pressure in Pa when given surface tension capillary radius [m] and wetting angle [rad]"""
        return 2*ts*np.cos(wetting_angle)/rbm  # capillary action [Pa]

    def Get_Kg_isentropic(g, Pg, sbm=0.0):
        """Return isentropic gas bulk modulus - pressures are absolute, sbm is capillary pressure"""
        return g.heat_cap*(Pg+sbm)

    def Get_Kg_isothermal(g, Pg, sbm=0.0):
        """Return isothermal gas bulk modulus - pressures are absolute, sbm is capillary pressure"""
        return Pg+sbm

    def Get_Cg_Koning(g, Pw):
        """Get the isothermal gas compressibility - pressures are absolute pressures"""
        return 1/Pw

    def Get_Cg_Koning_isentropic(g, Pw):
        """Get the isentropic gas compressibility - pressures are absolute pressures"""
        return 1/(g.heat_cap*Pw)

    def Get_Cg_Boutonnier(g, Pw, sbm, pwv):
        """Get the isothermal gas compressibility - pressures are absolute pressures, sbm is capillary pressure, pwv is water vapour pressure"""
        return 1/(Pw+sbm+pwv)

    def Get_Cg_Boutonnier_isentropic(g, Pw, sbm, pwv):
        """Get the isentropic gas compressibility - pressures are absolute pressures, sbm is capillary pressure, pwv is water vapour pressure"""
        return 1/(g.heat_cap*(Pw+sbm+pwv))

    