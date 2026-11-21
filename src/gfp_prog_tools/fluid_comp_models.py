# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:46:49 2021

@author: g.flood-page
"""
import sys  # nopep8
sys.path.append("D:\\DATA\\g.flood-page\\Documents\\Professionnel\\PhD\\11_Python_scripts")  # nopep8
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # color palettes
from scipy.optimize import fsolve, bisect, root_scalar

# returns fluid compressibility for each model
col_smoothing = sns.color_palette("husl", 8, as_cmap=False)  # set color scheme

cf_data = {  # {Label, {Oficial name, Colour}}
    "Koning": ["Koning (1963)", "blue"],
    "Koning_isentropic": ["Koning (1963) - isentropic", "steelblue"],
    "Hill": ["Hill (1965)", "orange"],
    "Fredlund": ["Fredlund et al. (2012)", "maroon"],
    "Schuurman_GFP": ["Schuurman (1966) - simplified", "orange"],
    "Schuurman": ["Schuurman (1966) - Original", "green"],
    "Schuurman_sat_pressure": ["Schuurman (1966) - With saturation pressure", "mediumseagreen"],
    "Schuurman_GFP_nosuc": ["Schuurman (1966) - GFP - No suction", "mediumaquamarine"],
    "Schuurman_nosuc": ["Schuurman (1966) - Original - No suction", "yellowgreen"],
    "Schuurman_nodis": ["Schuurman (1966) - Original - No dissolution", "darkgreen"],
    "Coussy": ["Coussy (2010) - Original", "violet"],
    "Coussy_nodis": ["Coussy (2010) - \u03C1f0\u2248\u03C1f - No dissolution", "darkviolet"],
    "Coussy_GFP_rho": ["Coussy (2010) - \u03C1f0\u2260\u03C1f", "purple"],
    "Coussy_GFP_Sr": ["Coussy (2010) - Sr0\u2260Sr", "fuchsia"],
    "Coussy_GFP_Sr_nodis": ["Coussy (2010) - Sr0\u2260Sr - h=0", "slateblue"],
    "Coussy_GFP_Sr_hyper": ["Coussy (2010) - Sr0\u2260Sr - Hyperbolic smoothing", col_smoothing[0]],
    "Coussy_GFP_Sr_log5": ["Coussy (2010) - Sr0\u2260Sr - Logarithmic smoothing - \u03B7=5", col_smoothing[1]],
    "Coussy_GFP_Sr_log10": ["Coussy (2010) - Sr0\u2260Sr - Logarithmic smoothing - \u03B7=10", col_smoothing[2]],
    "Coussy_GFP_Sr_log50": ["Coussy (2010) - Sr0\u2260Sr - Logarithmic smoothing - \u03B7=50", col_smoothing[3]],
    "Coussy_GFP_Sr_log100": ["Coussy (2010) - Sr0\u2260Sr - Logarithmic smoothing - \u03B7=100", col_smoothing[4]],
    "Coussy_GFP_Sr_log200": ["Coussy (2010) - Sr0\u2260Sr - Logarithmic smoothing - \u03B7=200", col_smoothing[5]],
    "Coussy_GFP_Sr_log400": ["Coussy (2010) - Sr0\u2260Sr - Logarithmic smoothing - \u03B7=400", col_smoothing[6]],
    "Coussy_GFP_Sr_log800": ["Coussy (2010) - Sr0\u2260Sr - Logarithmic smoothing - \u03B7=800", col_smoothing[7]],
    "Coussy_GFP_Sr_Srbloc": ["Coussy (2010) - Sr0\u2260Sr - Sr\u22641", "mediumorchid"],
    "Egis_Srunbloc": ["Boutonnier et al. (2019) - Unblocked Sr", "salmon"],
    "Egis": ["Boutonnier et al. (2019) - Original", "brown"],
    "Egis_novap": ["Boutonnier et al. (2019) - No vapour pressure", "sandybrown"],
    "Egis_nodis": ["Boutonnier et al. (2019) - h=0", "maroon"],
    "Egis_nosuc": ["Boutonnier et al. (2019) - No suction", "saddlebrown"],
    "Egis_nosuc_novap": ["Boutonnier et al. (2019) - No suction - No water vapour", "sandybrown"],
    "Saturated": ["Fully saturated", "black"]
}


def Get_Cf(Name, Cw, Sre, pw, h, sbm, Srg, pwg, Pref, pwv, pwe, SrFix, colour="black"):
    """Function will retreive the fluid compressibility corresponding to the Name provided using the parameters above"""
    assert Name in cf_data, "Fluid compressibility model not recognised"
    if Name == "Koning":
        return C_Koning(Cw, Sre, pw+Pref, SrFix)
    elif Name == "Koning_isentropic":
        return C_Koning_isentropic(Cw, Sre, pw+Pref, SrFix)
    if Name == "Hill":
        return C_Hill(Cw, Sre, pw+Pref, SrFix)
    elif Name == "Fredlund":
        return C_Fredlund(Cw, Sre, pw+Pref, h, sbm, SrFix)
    elif Name == "Schuurman_GFP":
        return C_Schuurman_gfp(Sre, pw, h, sbm, pwe, Pref)
    elif Name == "Schuurman":
        return C_Schuurman(Sre, pw, h, sbm, pwe, Pref)
    elif Name== "Schuurman_sat_pressure":
        return C_Schuurman_sat_pressure(Sre, pw, h, sbm, pwe, Pref, Cw)
    elif Name == "Schuurman_nodis":
        return C_Schuurman(Sre, pw, 0.0, sbm, pwe, Pref)
    elif Name == "Schuurman_GFP_nosuc":
        return C_Schuurman_gfp_nosuc(Sre, pw, h, pwe, Pref)
    elif Name == "Schuurman_nosuc":
        return C_Schuurman_nosuc(Sre, pw, h, pwe, Pref)
    elif Name == "Coussy":
        return C_Coussy(Cw, Sre, Pref+pw, h, Pref+pwe, SrFix)
    elif Name == "Coussy_GFP_Sr":
        return C_Coussy_GFP_Sr(Cw, Sre, Pref+pw, h, Pref+pwe, SrFix)
    elif Name == "Coussy_GFP_Sr_nodis":
        return C_Coussy_GFP_Sr(Cw, Sre, Pref+pw, 0.0, Pref+pwe, SrFix)
    elif Name == "Coussy_GFP_Sr_hyper":
        return C_Coussy_GFP_Sr_hyper(Cw, Sre, Pref+pw, h, Pref+pwe, SrFix)
    elif Name == "Coussy_GFP_Sr_log5":
        return C_Coussy_GFP_Sr_log(Cw, Sre, Pref+pw, h, Pref+pwe, 5, SrFix)
    elif Name == "Coussy_GFP_Sr_log10":
        return C_Coussy_GFP_Sr_log(Cw, Sre, Pref+pw, h, Pref+pwe, 10, SrFix)
    elif Name == "Coussy_GFP_Sr_log50":
        return C_Coussy_GFP_Sr_log(Cw, Sre, Pref+pw, h, Pref+pwe, 50, SrFix)
    elif Name == "Coussy_GFP_Sr_log100":
        return C_Coussy_GFP_Sr_log(Cw, Sre, Pref+pw, h, Pref+pwe, 100, SrFix)
    elif Name == "Coussy_GFP_Sr_log200":
        return C_Coussy_GFP_Sr_log(Cw, Sre, Pref+pw, h, Pref+pwe, 200, SrFix)
    elif Name == "Coussy_GFP_Sr_log400":
        return C_Coussy_GFP_Sr_log(Cw, Sre, Pref+pw, h, Pref+pwe, 400, SrFix)
    elif Name == "Coussy_GFP_Sr_log800":
        return C_Coussy_GFP_Sr_log(Cw, Sre, Pref+pw, h, Pref+pwe, 800, SrFix)
    elif Name == "Coussy_GFP_rho":
        return C_Coussy_GFP_rho(Cw, Sre, Pref+pw, h, Pref+pwe)
    elif Name == "Coussy_GFP_Sr_Srbloc":
        return C_Coussy_GFP_Sr_Srbloc(Cw, Sre, Pref+pw, h, Pref+pwe, SrFix)
    elif Name == "Coussy_nodis":
        return C_Coussy(Cw, Sre, Pref+pw, 0.0, Pref+pwe)
    elif Name == "Egis":
        return C_Egis(Cw, Sre, pw, h, sbm, Srg, pwg, Pref, pwv, SrFix)
    elif Name == "Egis_novap":
        return C_Egis(Cw, Sre, pw, h, sbm, Srg, pwg, Pref, 0.0, SrFix)
    elif Name == "Egis_nosuc":
        return C_Egis(Cw, Sre, pw, h, 0.0, Srg, pwg, Pref, 0.0, SrFix)
    elif Name == "Egis_nosuc_novap":
        return C_Egis(Cw, Sre, pw, h, 0.0, Srg, pwg, Pref, pwv, SrFix)
    elif Name == "Egis_nodis":
        return C_Egis(Cw, Sre, pw, 0.0, sbm, Srg, pwg, Pref, pwv, SrFix)
    elif Name == "Egis_Srunbloc":
        return C_Egis_Srunbloc(Cw, Sre, pw, h, sbm, Srg, pwg, Pref, pwv, SrFix)
    elif Name == "Saturated":
        return Saturated(Cw)

# ------------------------------------------------------------------------------
# Fully saturated - Unused
# ------------------------------------------------------------------------------


def Saturated(Cw):
    """Gives pure water bulk modulus supposing the soil is fully saturated 
    Variables: Pure water compressibility
    """
    return Cw, 1.0

# ------------------------------------------------------------------------------
#Series model (Wood/Koning/Reuss)
# ------------------------------------------------------------------------------
# as Sr is constant (Sr0=Sr) for Koning, gas pressure is taken constant and equal to

def C_Reuss(Cw, Sr, Cg, calculate_ci = False, Srci=[0,0]):
    """
    Reuss bound for fluid compressibility, supposes no dissolution
    If calculate_ci==True, returns also confidence interval
    Calculate ci supposing a ci on Sr and no error on other parameters
    If clculate_ci, can only compute results on a single value of Sr at a time
    """
    value =Sr*Cw + (1-Sr)*(Cg)
    if calculate_ci==False: 
        return value
    else:
        assert Srci[0]<Srci[1], "Sr Confidence intervals must be provided as [Srmin, Srmax]"
        mini = Srci[1]*Cw + (1-Srci[1])*(Cg)
        maxi =  Srci[0]*Cw + (1-Srci[0])*(Cg)
        return value, mini, maxi
# ------------------------------------------------------------------------------
#Parallel model (Hill/Voigt)
# ------------------------------------------------------------------------------
# as Sr is constant (Sr0=Sr) for Koning, gas pressure is taken constant and equal to

def C_Voigt(Cw, Sr, Cg, calculate_ci = False, Srci=[0,0]):
    """
    Voigt bound for fluid compressibility, supposes no dissolution
    If calculate_ci==True, returns also confidence interval
    Calculate ci supposing a ci on Sr and no error on other parameters
    If clculate_ci, can only compute results on a single value of Sr at a time
    """
    value =1/(Sr*1/Cw + (1-Sr)/(Cg))
    if calculate_ci==False: 
        return value
    else:
        assert Srci[0]<Srci[1], "Sr Confidence intervals must be provided as [Srmin, Srmax]"
        mini = 1/(Srci[0]*1/Cw + (1-Srci[0])/(Cg))
        maxi = 1/(Srci[1]*1/Cw + (1-Srci[1])/(Cg))
        return value, mini, maxi


# ------------------------------------------------------------------------------
#Series model with dissolution : Boutonnier model (Reuss)
# ------------------------------------------------------------------------------
# 

def C_series_diss(Cw, Sr, Cg, h):
    return Sr*Cw + (1-Sr+h*Sr)*(Cg)

# ------------------------------------------------------------------------------
#Update saturation ratio : calculate Sr from reference Sr0 and pf0
# ------------------------------------------------------------------------------
# 
def Update_Sr(heff, Sr0, Pref, pw0, pw, sbm, pwv=0.):
    """
    Calculate new saturation ratio at pressure pw on the basis of a reference point for saturation (Sr0, pw0)\n
    heff is the effective dissolution coefficient accounting for partial dissolution\n
    Pref is the reference pressure (atmospheric), sbm is capillary pressure. All in SI units\n
    Returns array or float according to input
    """
    return np.minimum(1, np.maximum(0, 1/(1-heff+(1-Sr0+heff*Sr0)/Sr0*(Pref+pw0+sbm-pwv)/(Pref+pw+sbm-pwv))))
    #bounded by 0 and 1

# ------------------------------------------------------------------------------
#Update saturation ratio : calculate Sr from reference Sr0 and pf0
# ------------------------------------------------------------------------------
# 
def Update_Sr_coussy_hypothesis(heff, Sr0, Pref, pw0, pw, sbm):
    """
    Calculate new saturation ratio at pressure pw on the basis of a reference point for saturation (Sr0, pw0)
    heff is the effective dissolution coefficient accounting for partial dissolution,
    Pref is the reference pressure (atmospheric), sbm is capillary pressure. All in SI units
    Returns array or float according to input
    Hypothesis Sr/Sr0 approx 1 added, to strictly respect Coussy hypothesis, sbm should be take as 0
    """
    return np.minimum(1, np.maximum(0, 1+heff*Sr0-(1-Sr0+heff*Sr0)*(Pref+pw0+sbm)/(Pref+pw+sbm)))
    #bounded by 0 and 1

# ------------------------------------------------------------------------------
# Long term fluid compressibility
# ------------------------------------------------------------------------------
def Get_Kf_long_term(Cw, Sr, Cg, heff):
    """
    Computes the fluid compressibility modulus regardless of saturation state (supposes that domain D3 is valid
    Input in SI units : Cw is water compressibility, Cg is gas comp, heff is effective solubility coefficient
    Sr is current saturation ratio (needs to be previously calculated)
    """
    return 1/(Sr*Cw + (1-Sr+heff*Sr)*(Cg))

# ------------------------------------------------------------------------------
# Long term fluid compressibility
# ------------------------------------------------------------------------------
def Get_Sr_Cf_logarithmic_smooth(Cw, heff, Sr0, pw, pw0, l=20., sc=0.0, Pref = 101325):
    """
    Computes saturation ratio and fluid compressibility modulus with log smoothing on x (mole fraction)
    Input in SI units : Cw is water compressibility, heff is effective solubility coefficient
    Sr0 is ini saturation ratio at pw0, pw is current pw, l is smoothing exponent, sc : capillary suction
    """
    Pg0 = Pref+pw0+sc
    Pgsat = Pg0+Pg0*(1-Sr0)/(heff*Sr0)  # Saturation pressure
    Pg = Pref+pw+sc
    Sr = -Pg*Sr0*l/(Pgsat*Sr0*heff*(np.log(np.exp(l)+np.exp(Pg0*l/Pgsat))-np.log(np.exp(l)+np.exp(Pg*l/Pgsat)))+Sr0*heff*l*(Pg-Pg0)-Sr0*l*(Pg-Pg0)-l*Pg0)
    Cf = Sr*Cw+(1-Sr)*1/Pg+Sr*heff/Pg*(1/(1+np.exp(-l*(Pgsat-Pg)/Pgsat)))
    return Cf, Sr

# ------------------------------------------------------------------------------
#Koning (1963)
# ------------------------------------------------------------------------------
# as Sr is constant (Sr0=Sr) for Koning, gas pressure is taken constant and equal to


def C_Koning(Cw, Sr, Pw, SrFix=-1):
    if SrFix != -1:  # Checks if the value of Sr is specified by the user
        Sr = SrFix
    return Sr*Cw + (1-Sr)/(Pw), Sr
# Simplest compressibility - series average

# ------------------------------------------------------------------------------
#Koning (1963) with isentropic conditions
# ------------------------------------------------------------------------------
# as Sr is constant (Sr0=Sr) for Koning, gas pressure is taken constant and equal to
#TODO

def C_Koning_isentropic(Cw, Sr, Pw, SrFix=-1, heat_cap = 1.4):
    if SrFix != -1:  # Checks if the value of Sr is specified by the user
        Sr = SrFix
    return Sr*Cw + (1-Sr)/(heat_cap*Pw), Sr
# Simplest compressibility - series average

# ------------------------------------------------------------------------------
#Hill (1965)
# ------------------------------------------------------------------------------
# as Sr is constant (Sr0=Sr) for Koning, gas pressure is taken constant and equal to


def C_Hill(Cw, Sr, Pw, SrFix=-1):
    if SrFix != -1:  # Checks if the value of Sr is specified by the user
        Sr = SrFix
    return 1/(Sr*1/Cw + (1-Sr)/(1/Pw)), Sr
# Simplest compressibility - parallel average

# ------------------------------------------------------------------------------
#Fredlund (2012)
# ------------------------------------------------------------------------------
# Sr is constant (Sr0=Sre=Sr)


def C_Fredlund(Cw, Sr, Pw, h, sbm, SrFix=-1):
    if SrFix != -1:  # Checks if the value of Sr is specified by the user
        Sr = SrFix
    return Sr*Cw + (1-Sr+h*Sr)/(Pw+sbm), Sr

# ------------------------------------------------------------------------------
# Schuurman (1966) : equation modified in order to be expressed as function of Sr
# ------------------------------------------------------------------------------
# This induces a slight error due to the fact that initial and final porosity n are supposed to be identical
# Sr0  #initial saturation ratio


def C_Schuurman_gfp(Sr0, pw, h, sbm0, pwe, Pref):
    def solve_Schu(Sr, *prm):
        return prm[0]-(1-Sr0+h*Sr)/(1-Sr+h*Sr)*Pg0-sbm0*((1-Sr0)/(1-Sr))**(1/3)
    Pw = Pref+pw
    Pg0 = Pref + pwe+sbm0
    tol = 10e-12
    if pw >= pwe:
        # Sri0 = fsolve(solve_Schu, x0=Sr0, xtol = tol, args=(Pw))
        # Sri2 = bisect(solve_Schu, a=0.1, b=1.0-1e-10, xtol = tol, args=(Pw)) #: should give same result
        Sri = root_scalar(solve_Schu, bracket=[1e-15, 1.0-1e-15], args=(Pw)).root
    else:
        Sri = root_scalar(solve_Schu, bracket=[1e-15, 1.0-1e-15], args=(Pref+pwe)).root
        #Sri = fsolve(solve_Schu, x0=Sr0, xtol = tol, args=(Pref+pwe))
        # calculate Sr for pwe water pressure
    return 1/(((1-Sr0+h*Sri)*Pg0)/((1-Sri+h*Sri)**2)-sbm0/(3*(1-Sri))*((1-Sr0)/(1-Sri))**(1/3)), Sri
    # pass #model not designed for pw<pwe

# ------------------------------------------------------------------------------
# Schuurman (1966) : equation modified in order to be expressed as function of Sr with no capillary suction
# ------------------------------------------------------------------------------
# This induces a slight error due to the fact that initial and final porosity n are supposed to be identical
# Sr0  #initial saturation ratio


def C_Schuurman_gfp_nosuc(Sr0, pw, h, pwe, Pref):
    def solve_Schu2(Sr, *prm):
        return prm[0]-(1-Sr0+h*Sr)/(1-Sr+h*Sr)*Pg0
    Pw = Pref+pw
    Pg0 = Pref + pwe
    if pw >= pwe:
        Sri = (Pw-(1-Sr0)*Pg0)/(Pw*(1-h)+h*Pg0)
        #Sri1 = root_scalar(solve_Schu2, bracket = [1e-15, 1.0-1e-15], args = (Pw)).root
    else:
        #Sri1 = root_scalar(solve_Schu2, bracket = [1e-15, 1.0-1e-15], args = (Pref+pwe)).root
        Sri = (Pref+pwe-(1-Sr0)*Pg0)/((Pref+pwe)*(1-h)+h*Pg0)  # calculate Sr for pwe water pressure
    if Sri >= 1:
        print(f"Beware, the saturations calculated using Schuurman_gfp_nosuc method for pw={pw} Pa is above 1 (Sr = {Sri})")
    return 1/(((1-Sr0+h*Sri)*Pg0)/((1-Sri+h*Sri)**2)), Sri
    # model not designed for pw<pwe

# ------------------------------------------------------------------------------
# Schuurman : original version
# ------------------------------------------------------------------------------


def C_Schuurman(Sr0, pw, h, sbm0, pwe, Pref):
    def solve_Schu1(Vg, *prm):
        return prm[0]-(Vg0+Vd)/(Vg+Vd)*Pg0-sbm0*(Vg0/Vg)**(1/3)
    Pw = Pref+pw
    Pg0 = Pref + pwe+sbm0
    #tol = 10e-12
    Vf0 = 1.0  # fluid volume : changing this value has no influence on the result
    Vg0 = Vf0*(1-Sr0)  # initial gas volume
    Vw = Vf0*Sr0  # liquid volume
    Vd = h*Vw  # dissolved gas volume
    # if pw >= pwe:
    Vgi = root_scalar(solve_Schu1, bracket=[Vf0*1e-15, Vf0*1.0-1e-15], args=(Pw)).root
        #Vgi = fsolve(solve_Schu1, x0=Vg0, xtol=tol, args = (Pw))
        #Vgi2[i] = bisect(solve_Schu1, a=0.00001, b=Vf0, xtol = tol, args = (Pw))
    # else:
    #     #Vgi = fsolve(solve_Schu1, x0=Vg0, xtol=tol, args = (Pref+pwe))
    #     Vgi = root_scalar(solve_Schu1, bracket=[Vf0*1e-15, Vf0*1.0-1e-15], args=(Pref+pwe)).root
    return 1/(Vgi+Vw)/(((Vg0+Vd)*Pg0)/((Vgi+Vd)**2)-sbm0/3*1/Vgi*(Vg0/Vgi)**(1/3)), 1-Vgi/Vf0
    # model not designed for pw<pwe

# ------------------------------------------------------------------------------
# Schuurman : original version with a dissolved gas saturation pressure
# ------------------------------------------------------------------------------


def C_Schuurman_sat_pressure(Sr0, pw, h, sbm0, pwe, Pref, Cw):
    def solve_Schu1(Vg, *prm):
        return prm[0]-(Vg0+Vd)/(Vg+Vd)*Pg0-sbm0*(Vg0/Vg)**(1/3)
    Pw = Pref+pw
    Pg0 = Pref + pwe+sbm0
    #tol = 10e-12
    Vf0 = 1.0  # fluid volume : changing this value has no influence on the result
    Vg0 = Vf0*(1-Sr0)  # initial gas volume
    Vw = Vf0*Sr0  # liquid volume
    Vd = h*Vw  # dissolved gas volume
    if pw>(Pg0*(Vg0+Vd)/Vd): #soil fully saturated
        return Cw, 1
    elif pw >= pwe:
        Vgi = root_scalar(solve_Schu1, bracket=[Vf0*1e-15, Vf0*1.0-1e-15], args=(Pw)).root
        #Vgi = fsolve(solve_Schu1, x0=Vg0, xtol=tol, args = (Pw))
        #Vgi2[i] = bisect(solve_Schu1, a=0.00001, b=Vf0, xtol = tol, args = (Pw))
    else:
        #Vgi = fsolve(solve_Schu1, x0=Vg0, xtol=tol, args = (Pref+pwe))
        Vgi = root_scalar(solve_Schu1, bracket=[Vf0*1e-15, Vf0*1.0-1e-15], args=(Pref+pwe)).root
    return 1/(Vgi+Vw)/(((Vg0+Vd)*Pg0)/((Vgi+Vd)**2)-sbm0/3*1/Vgi*(Vg0/Vgi)**(1/3)), 1-Vgi/Vf0
    # model not designed for pw<pwe

# ------------------------------------------------------------------------------
# Schuurman : original version disregarding capillary effects
# ------------------------------------------------------------------------------


def C_Schuurman_nosuc(Sr0, pw, h, pwe, Pref):
    def solve_Schu(Vg, *prm):
        return prm[0]-(Vg0+Vd)/(Vg+Vd)*Pg0
    Pw = Pref+pw
    Pg0 = Pref + pwe
    #tol = 10e-12
    Vf0 = 1.0  # fluid volume : changing this value has no influence on the result
    Vg0 = Vf0*(1-Sr0)  # initial gas volume
    Vw = Vf0*Sr0  # liquid volume
    Vd = h*Vw  # dissolved gas volume
    if pw >= pwe:
        Vgi = (Vg0+Vd)*Pg0/Pw-Vd
        #Vgi1 = root_scalar(solve_Schu, bracket = [Vf0*1e-15, Vf0*1.0-1e-15], args = (Pw)).root
    else:
        Vgi = (Vg0+Vd)*Pg0/(Pref+pwe)-Vd
        #Vgi1 = root_scalar(solve_Schu, bracket = [Vf0*1e-15, Vf0*1.0-1e-15], args = (Pref+pwe)).root
    if Vgi < 0:
        print(f"Beware, the gas volume calculated using Schuurman_nosuc method for pw={pw} Pa is negative (Vgi = {Vgi}m3) - this is equivalent to saying that Sr>1")
    return 1/(Vgi+Vw)/(((Vg0+Vd)*Pg0)/((Vgi+Vd)**2)), 1-Vgi/Vf0
    # model not designed for pw<pwe

# ------------------------------------------------------------------------------
#Coussy (2010)
# ------------------------------------------------------------------------------
# Capillary effects neglected


def C_Coussy(Cw, Sr0, Pw, h, Pw0, SrFix = -1):
    if SrFix == -1:  # Do not want Sr to be imposed
        Sr = 1+h*Sr0-(1-Sr0+h*Sr0)*Pw0/Pw
        if Sr>1: #fully saturated
            return Cw, 1
    else:  # Want to fix evolution of Sr
        Sr = SrFix
    
    # Cf = Sr_Sr0*((1-Sr0)*Pw0/Pw*(1/Pw-Cw)+h*Sr0*(Cw+Pw0/Pw*(1/Pw-Cw)))+Cw #identical to other
    return (1-Sr0+Sr0*h)/((Pw)**2/(Pw0))+Sr0*Cw, Sr 

# ------------------------------------------------------------------------------
# Coussy (2010) without hypothesis that fluid density barely changes rhof!= rhof0
# ------------------------------------------------------------------------------
# Capillary effects neglected


def C_Coussy_GFP_rho(Cw, Sr0, Pw, h, Pw0):
    rhof_rhof0 = 1/(Sr0 + (1-Sr0)*Pw0/Pw + h*Sr0*(Pw0-Pw)/Pw)
    Sr_Sr0 = rhof_rhof0
    Sr = Sr0*Sr_Sr0
    return rhof_rhof0*(1-Sr0+Sr0*h)/((Pw)**2/(Pw0))+Sr0*Cw, Sr

# ------------------------------------------------------------------------------
# Coussy (2010) - GFP - nosuc : Coussy recalculated using Egis-3SR technique (dSR/dp) without assumption
# ------------------------------------------------------------------------------
# Capillary effects neglected


def C_Coussy_GFP_Sr(Cw, Sr0, Pw, h, Pw0, SrFix=-1):
    if SrFix == -1:  # Do not want Sr to be imposed
        Sr_Sr0 = 1/(Sr0 + Pw0/Pw*(1-Sr0)-h*Sr0*(Pw-Pw0)/Pw)  # Sr/Sr0
        Sr = Sr_Sr0*Sr0  # get Saturation ratio
        if Sr>1: #fully saturated
            return Cw, 1
    else:  # Want to fix evolution of Sr
        Sr = SrFix
        Sr_Sr0 = Sr/Sr0
    Cf = Sr_Sr0*((1-Sr0)*Pw0/Pw*(1/Pw-Cw)-h*Sr0*(Pw-Pw0)/Pw*(1/Pw-Cw)+h*Sr0/Pw)+Cw
    # Cf = Sr_Sr0*((1-Sr0)*Pw0/Pw*(1/Pw-Cw)+h*Sr0*(Cw+Pw0/Pw*(1/Pw-Cw)))+Cw #identical to other
    return Cf, Sr

# ------------------------------------------------------------------------------
# Coussy (2010) - GFP - nosuc - Hyperbolic dissolution function : Coussy recalculated using Egis-3SR technique (dSR/dp) without assumption
# ------------------------------------------------------------------------------
# Capillary effects neglected


def C_Coussy_GFP_Sr_hyper(Cw, Sr0, Pw, h, Pw0, SrFix=-1):
    if SrFix == -1:  # Do not want Sr to be imposed
        Pgsat = Pw0+Pw0*(1-Sr0)/(h*Sr0)  # Saturation pressure
        Sr_Sr0 = 1/(Sr0+Pw0/Pw*(1-Sr0)-h*Sr0*Pgsat/Pw*(math.tanh(Pw/Pgsat)-math.tanh(Pw0/Pgsat)))
        Sr = Sr_Sr0*Sr0  # get Saturation ratio
    else:  # Want to fix evolution of Sr
        Sr = SrFix
        Sr_Sr0 = Sr/Sr0

    Cf = Cw + Sr_Sr0*((1-Sr0)*Pw0/Pw*(1/Pw-Cw) - h*Sr0/Pw*Pgsat*(math.tanh(Pw/Pgsat)-math.tanh(Pw0/Pgsat))*(1/Pw-Cw) + h*Sr0/Pw*(1-(math.tanh(Pw/Pgsat))**2))
    return Cf, Sr

# ------------------------------------------------------------------------------
# Coussy (2010) - GFP - nosuc - Logarithmic dissolution function : Coussy recalculated using Egis-3SR technique (dSR/dp) without assumption
# ------------------------------------------------------------------------------
# Capillary effects neglected
# eta is a fitting parameter


def C_Coussy_GFP_Sr_log(Cw, Sr0, Pw, h, Pw0, eta, SrFix=-1):
    if SrFix == -1:  # Do not want Sr to be imposed
        Pgsat = Pw0+Pw0*(1-Sr0)/(h*Sr0)  # Saturation pressure
        ePw = np.exp(-eta*(Pgsat-Pw)/Pgsat)  # exponential containing Pw
        ePw0 = np.exp(-eta*(Pgsat-Pw0)/Pgsat)  # exponential containing Pw0
        Sr_Sr0 = 1/(Sr0+(1-Sr0)*Pw0/Pw - Sr0/Pw*h*(Pw-Pw0+Pgsat/eta*np.log((1+ePw0)/(1+ePw))))
        #Sr_Sr0 = 1/(Sr0+(Pw0/Pw)*(1-Sr0)-Sr0/Pw*h*(Pw-Pw0+Pgsat/eta*math.log((1+ePw0)/(1+ePw))))
        Sr = Sr_Sr0*Sr0  # get Saturation ratio
    else:  # Want to fix evolution of Sr
        Sr = SrFix
        Sr_Sr0 = Sr/Sr0
    Cf = Cw + Sr_Sr0*((1-Sr0)*Pw0/Pw*(1/Pw-Cw) - Sr0/Pw*h*(Pw-Pw0+Pgsat/eta*np.log((1+ePw0)/(1+ePw)))*(1/Pw-Cw) + Sr0/Pw*h*(1-ePw/(1+ePw)))  # real
    return Cf, Sr

# ------------------------------------------------------------------------------
# Coussy (2010) - GFP - nosuc : Coussy recalculated using Egis-3SR technique (dSR/dp) without assumption
# Sr is blocked at 1 and Cf is imposed at Cw if Sr=1
# ------------------------------------------------------------------------------
# Capillary effects neglected


def C_Coussy_GFP_Sr_Srbloc(Cw, Sr0, Pw, h, Pw0, SrFix=-1):
    if SrFix == -1:  # Do not want Sr to be imposed
        Sr_Sr0 = 1/(Sr0 + Pw0/Pw*(1-Sr0)-h*Sr0*(Pw-Pw0)/Pw)  # Sr/Sr0
        Sr_Sr0 = min(Sr_Sr0, 1/Sr0)
        Sr = Sr_Sr0*Sr0  # get Saturation ratio
    else:  # Want to fix evolution of Sr
        Sr = SrFix
        Sr_Sr0 = Sr/Sr0
    if Sr >= 1:
        Cf = Cw
    else:
        Cf = Sr_Sr0*(Pw0/(Pw**2)*(1-Sr0)+Pw0/Pw*(1-Sr0)*Cw-h*Sr0*(Cw*(Pw-Pw0)/Pw-Pw0/(Pw**2)))+Cw
    return Cf, Sr


# ------------------------------------------------------------------------------
# Egis-3SR
# ------------------------------------------------------------------------------
# supposes a Sre, pwe, Srg and pwg : pw is variable
def C_Egis_Srunbloc(Cw, Sre, pw, h, sbm, Srg, pwg, Pref, pwv, SrFix=-1):
    assert Sre >= Srg, "Sre<Srg : please change Sre or Srg"
    if pw < 0:
        if SrFix == -1:  # Do not want Sr to be imposed
            Sr = Sre-pw/pwg*(Sre-Srg)
        else:
            Sr = SrFix
        return -1/(Sr*pwg)*(Sre-Srg)+Cw, Sr
    else:
        if SrFix == -1:  # Do not want Sr to be imposed
            Sr = 1/(1-h+(1-Sre+h*Sre)/Sre*(Pref+sbm+0-pwv)/(Pref+sbm+pw+pwv))
        else:
            Sr = SrFix
        return (1-Sr+h*Sr)/(Pref+sbm+pw-pwv)+Cw, Sr

# ------------------------------------------------------------------------------
# Egis-3SR - the saturation ratio is blocked at 1
# ------------------------------------------------------------------------------
# supposes a Sre, pwe, Srg and pwg : pw is variable


def C_Egis(Cw, Sre, pw, h, sbm, Srg, pwg, Pref, pwv, SrFix=-1):
    assert Sre >= Srg, "Sre<Srg : please change Sre or Srg"
    if pw < 0:
        if SrFix == -1:  # Do not want Sr to be imposed
            Sr = min(Sre-pw/pwg*(Sre-Srg), 1.0)
        else:
            Sr = SrFix
        if Sr >= 1:
            Cf = Cw
        else:
            Cf = -1/(Sr*pwg)*(Sre-Srg)+Cw
    else:
        if SrFix == -1:  # Do not want Sr to be imposed
            Sr = min(1/(1-h+(1-Sre+h*Sre)/Sre*(Pref+sbm+0-pwv)/(Pref+sbm+pw+pwv)), 1)
        else:
            Sr = SrFix
        if Sr >= 1:
            Cf = Cw
        else:
            Cf = (1-Sr+h*Sr)/(Pref+sbm+pw-pwv)+Cw
    return Cf, Sr
