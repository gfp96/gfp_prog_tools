# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:41:16 2021

@author: g.flood-page
"""

# File containing soil class and sub classes for different soils
import numpy as np
# ---------------------------- Soil Class---------------------------------------


class Soil:
    """ Defines general properties of ss"""

    def __init__(s, name, Id):
        s.name = name
        s.Id=Id

    def Reset(s):
        s.Id = 10000.0  # density index
        s.emax = 0.0  # Max1void index
        s.emin = 0.0  # Min void index
        s.e0 = 0.0
        s.e = 0.0 #void index
        s.n0 = 0.0
        s.n #current porosity
        s.rhos = 2.7e3  # specific mass of the grains [kg/m3]
        s.rhod = 0.0 #Dry mass density
        s.rho = 0.0 #Real mass density
        s.poisson = -1.0  # poisson coefficient
        s.G0 = 0.0  # small strain modulus
        
        

    Id = 0.0  # density index
    emax = 0.0  # Max1void index
    emin = 0.0  # Min void index
    e0 = 0.0
    Ks = 3.6e10 #bulk modulus grains
    e = -1
    n0 = -1
    n = -1
    Cu = 0.0 #granular soi uniformity
    D50 = -1 #median grain size
    rhos = 2.7e3  # specific mass of the grains [kg/m3]
    poisson = -1.0  # poisson coefficient
    G0 = 0.0  # small strain modulus
    rhod = 0.0  # Dry mass density
    rho = 0.0 #Real mass density
    data_origin = "N/A"  # unknown origin of the data
    #error margins
    err_e = -1 #error on e 
    err_rhod = -1 #error on rhod
    err_n = -1 # error on n

    def Get_e0(s):  # get initial void index #checked
        """get initial void index"""
        s.e0 = -s.Id*(s.emax-s.emin)+s.emax
        return s.e0

    def Get_n(s, e):  # get porosity from void index #checked
        """get porosity from void index"""
        return e/(1+e)

    def Get_e(s, n):  # get void index from porosity #checked
        """get void index from porosity"""
        return n/(1-n)
    
    def Get_Id(s, e):  # get void index from porosity #checked
        """get density index from void ratio"""
        return (s.emax-e)/(s.emax-s.emin)

    def Get_Ku(s, Cf, Cs, Ck, n, calculate_ci=False, Cfci=[0,0], Ckci=[0,0], nci=[0,0]):  # checked
        """
        Get undrained bulk modulus from Cf, Cs, Ck and n
        If calculate_ci, also returns confidence interval calculated based on
        ci from Cf, Ck and n : supposes no error on Cs
        ci must be provided as [min, max]
        When calculating ci : can only compute one value of Ku at a time
        """
        value = 1/((n*Cf+(1-n)*Cs-Cs**2/Ck)/(1+n*Cf/Ck-(1+n)*Cs/Ck)) 
        if calculate_ci==False:
            return value
        else:
            # assert Cfci[0]<Cfci[1] and Ckci[0]<Ckci[1]
            combs = np.zeros((2**3, 3), dtype = float)
            counter = 0
            for i, Cfval in enumerate(Cfci):
                for j, nval in enumerate(nci):
                    for k, Kval in enumerate(Ckci):
                        combs[counter, 0] = Cfval
                        combs[counter, 1] = nval
                        combs[counter, 2] = Kval
                        counter+=1
            calc = 1/((combs[:, 1]*combs[:, 0]+(1-combs[:, 1])*Cs-Cs**2/combs[:, 2])/(1+combs[:, 1]*combs[:, 0]/combs[:, 2]-(1+combs[:, 1])*Cs/combs[:, 2]))  
            mini = np.nanmin(calc, axis = 0)
            maxi = np.nanmax(calc, axis = 0)
            return value, mini, maxi


    def Get_B(s, Cf, Ck, n, Cs = 0, compressible_grains=False):  # checked
        """Get Skempton's B value"""
        if compressible_grains == True:
            return 1/(1+n*(Cf-Cs)/(Ck-Cs))
        else:
            return 1/(1+n*(Cf)/(Ck))

    def Get_K(s, G, poisson, calculate_ci = False, Gmin = 0, Gmax = 0, dpois = 0):  # checked
        """
        Gets the bulk modulus from corresponding shear modulus and poisson ratio
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined before
        Supposes that highers error bound given by pe+dpe and lower by pe-dpe
        """
        #tested and validated with error calculation
        # Pref usually taken as atmospheric pressure
        value = 2*G*(1+poisson)/(3*(1-2*poisson))  # drained bulk modulus 
        if calculate_ci==False:
            return  value
        else:
            maxi = np.nanmax(np.array([2*Gmax*(1+(poisson+dpois))/(3*(1-2*(poisson+dpois))), 2*Gmax*(1+(poisson-dpois))/(3*(1-2*(poisson-dpois)))]), axis = 0)
            mini = np.nanmin(np.array([2*Gmin*(1+(poisson+dpois))/(3*(1-2*(poisson+dpois))), 2*Gmin*(1+(poisson-dpois))/(3*(1-2*(poisson-dpois)))]), axis = 0)
            return np.array([value, mini, maxi], dtype = float)
    
    def Get_G(s, K, poisson):
        """Calculate shear modulus from bulk modulus and poisson ratio. No uncertainty calculation is implemented"""
        return 3*K*(1-2*poisson)/(2*(1+poisson)) 

    def Get_M_from_G_K(s, G, K, calculate_ci=False, Gci=[0,0], Kci=[0,0]):
        """
        Gets the bulk modulus from corresponding shear modulus and bulk modulus
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined before
        """
        value = K+4/3*G   
        if calculate_ci==False:
            return  value
        else:
            assert Kci[0]<=Kci[1] and Gci[0]<=Gci[1], "Please check that Kci[0]<Kci[1] and Gci[0]<Gci[1]"
            mini = Kci[0]+4/3*Gci[0]
            maxi = Kci[1]+4/3*Gci[1]
            return np.array([value, mini, maxi], dtype = float)

    
    def Get_E_from_G_K(s, G, K, calculate_ci=False, Gci=[0,0], Kci=[0,0]):
        """
        Gets the Young modulus from corresponding shear modulus and bulk modulus
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined in additional parameters
        """
        value =  9*K*G/(3*K+G)
        if calculate_ci==False:
            return value
        else:
            combs = np.zeros((2**2, 2), dtype = float)
            counter = 0
            for i, Gval in enumerate(Gci):
                for j, Kval in enumerate(Kci):
                    combs[counter, 0] = Gval
                    combs[counter, 1] = Kval
                    counter+=1
            calc = 9*combs[:, 1]*combs[:, 0]/(3*combs[:, 1]+combs[:, 0])
            mini = np.nanmin(calc, axis = 0)
            maxi = np.nanmax(calc, axis = 0)
            return value, mini, maxi

    def Get_E_from_G_poisson(s, G, poisson, calculate_ci=False, Gci=[0,0], poisci=[0,0]):
        """
        Gets the Young modulus from corresponding shear modulus and bulk modulus
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined in additional parameters
        !!! ci calculation has not been tested !!!
        """
        value = 2*G*(1+poisson)
        if calculate_ci==False:
            return value
        else:
            combs = np.zeros((2**2, 2), dtype = float)
            counter = 0
            for i, Gval in enumerate(Gci):
                for j, poisval in enumerate(poisci):
                    combs[counter, 0] = Gval
                    combs[counter, 1] = max(-1, min(0.499999, poisval))
                    counter+=1
            calc = 2*combs[:, 0]*(1+combs[:, 1])
            mini = np.nanmin(calc, axis = 0)
            maxi = np.nanmax(calc, axis = 0)
            return value, mini, maxi


    def Get_K_from_G_M(s, G, M, calculate_ci=False, Gci=[0,0], Mci=[0,0]):
        """
        Gets bulk modulus from shear and oedometric moduli - G and M arrays must have same length
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined before. Can only be used on a single value (not an array)
        """
        value = M-4*G/3
        if calculate_ci==False:
            return  value
        else:
            assert Mci[0]<=Mci[1] and Gci[0]<=Gci[1], "Please check that Mci[0]<Mci[1] and Gci[0]<Gci[1]"
            mini = Mci[0]-4/3*Gci[1]
            maxi = Mci[1]-4/3*Gci[0]
            return np.array([value, mini, maxi], dtype = float)
        
    
    def Get_K_from_G_E(s, G, E,  calculate_ci=False, Gci=[0,0], Eci=[0,0], Kmin=1e5):
        """
        Gets bulk modulus from shear and Young moduli - G and E arrays must have same length
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined before. Can only be used on a single value (not an array)
        """
        value = E*G/(3*(3*G-E))
        if calculate_ci==False:
            return  value
        else:
            combs = np.zeros((2**2, 2), dtype = float)
            counter = 0
            for i, Gval in enumerate(Gci):
                for j, Eval in enumerate(Eci):
                    combs[counter, 0] = Gval
                    combs[counter, 1] = Eval
                    counter+=1
            calc = combs[:, 1]*combs[:, 0]/(3*(3*combs[:, 0]-combs[:, 1]))
            mini = np.nanmin(calc, axis = 0)
            maxi = np.nanmax(calc, axis = 0)
            return value, mini, maxi

        
    
    def Get_M_from_G_pois(s, G, pois):
        """Gets P-wave modulus from shear modulus and poisson ratio - G and poisson arrays must have same length, or poisson is constant"""
        return 2*G*(1-pois)/(1-2*pois)

    def Get_Sr_from_B(s, B, K, n, Cg, Cw, Cf_model = "Koning"):
        """Returns the Saturationn ratio using fluid compressibility"""
        if Cf_model == "Koning": #simplest
            return (1/(n*K)*(1/B-1)-Cg)/(Cw-Cg)
        else:
            return 1

    def Get_rho_from_Sr(s, n, Sr, rhow): 
        """Returns the density of the soil"""
        return s.rhod+n*Sr*rhow
    
    def Get_rhos_from_Srs(s, Sr, n, rhow): 
        """Returns the density of the soil - duplicate : to be deprecated"""
        return s.rhod+n*Sr*rhow

    def Set_rhod_from_n(s):
        """Returns the dry density of the soil. Soil porosity must be previously defined"""
        s.rhod = (1-s.n)*s.rhos
    
    def Get_rhod_from_n(s, n):
        """Returns the dry density of the soil. Soil porosity must be previously defined"""
        return (1-n)*s.rhos

    def Get_rhod_from_e(s, e):
        """Returns the dry density of the soil. Void ratio(s) do not have to be previously initialised"""
        return s.rhos/(1+e)
    
    def Get_e_from_e0_epsv(s, e0, epsv):
        """Returns the current void ratio from initial e0 and current volumetric strain (with reference to initial state)"""
        return e0-(1+e0)*epsv
    
    def Get_modulus_from_vel(s, vel, calculate_ci = False, rhoci = [0,0], velci = [0,0]):
        """
        Returns soil modulus from wave velocity
        If calculate_ci : returns also the confidence interval
        Need to provide ci for rho and vel : [mini, maxi]
        """
        value = s.rho*vel**2
        if calculate_ci==False:
            return value
        else:
            assert rhoci[0]<=rhoci[1] and velci[0]<=velci[1], "Please check that rhoci[0]<rhoci[1] and velci[0]<velci[1]"
            mini = rhoci[0]*velci[0]**2
            maxi = rhoci[1]*velci[1]**2
            return value, mini, maxi
    
    def Get_vel_from_modulus(s, mod,  calculate_ci = False, rhoci = [0,0], modci = [0,0]):
        """
        Returns wave velocity from soil modulus. Soil density must be defined in soil class
        If calculate_ci : returns also the confidence interval
        Need to provide ci for rho and vel : [mini, maxi]
        """
        value = np.sqrt(mod/s.rho)
        if calculate_ci==False:
            return value
        else:
            assert rhoci[0]<=rhoci[1] and modci[0]<=modci[1], "Please check that rhoci[0]<rhoci[1] and modci[0]<modci[1]"
            mini = np.sqrt(modci[0]/rhoci[1])
            maxi = np.sqrt(modci[1]/rhoci[0])
            return value, mini, maxi
        

    def Get_Cf_from_B(s, B, n, K, Cs=0, calculate_ci = False, 
        ciB = [0,0], cin = [0,0], ciK = [0,0], minCf = 4.19e-10): #verified 06/10/2022
        """"
        Returns fluid compressibility from B value (supposing incompressible grains if Cs=0)
        If calculate_ci = True : calculate resulting confidence interval from ciB, cin and ciK
        For ci interval calculation : B must be a single value, not an array
        maxi value of Cf is 1/Patm
        """
        if Cs==0:
            value = (1/B-1)/(n*K)
        else:
            value = Cs+(1/K-Cs)/n*(1/B-1)
        if calculate_ci==False:
            return value
        else:
            combs = np.zeros((2**3, 3), dtype = float)
            counter = 0
            for i, Bval in enumerate(ciB):
                for j, nval in enumerate(cin):
                    for k, Kval in enumerate(ciK):
                        combs[counter, 0] = Bval
                        combs[counter, 1] = nval
                        combs[counter, 2] = Kval
                        counter+=1
            calc = Cs+(1/combs[:, 2]-Cs)/combs[:, 1]*(1/combs[:, 0]-1)
            #calc = np.where(calc<0, maxKf, calc) # replace negative Cf by Cf of pure water
            mini = max(minCf, np.nanmin(calc, axis = 0))
            maxi = max(minCf, np.nanmax(calc, axis = 0))
            return value, mini, maxi

    def Get_Kf_from_B(s, B, n, K, Cs=0, calculate_ci = False, 
        ciB = [0,0], cin = [0,0], ciK = [0,0], minKf=1e5, maxKf = 2.4e9): #verified 06/10/2022
        """"
        Returns fluid bulk modulus from B value (supposing incompressible grains if Cs=0)
        If calculate_ci = True : calculate resulting confidence interval from ciB, cin and ciK
        For ci interval calculation : B must be a single value, not an array
        ci calculation only programmed with Cs different from 0
        """
        if Cs==0:
            value = 1/((1/B-1)/(n*K))
        else:
            value = 1/(Cs+(1/K-Cs)/n*(1/B-1))
        if calculate_ci==False:
            return value
        else:
            combs = np.zeros((2**3, 3), dtype = float)
            counter = 0
            for i, Bval in enumerate(ciB):
                for j, nval in enumerate(cin):
                    for k, Kval in enumerate(ciK):
                        combs[counter, 0] = Bval
                        combs[counter, 1] = nval
                        combs[counter, 2] = Kval
                        counter+=1
            calc = 1/(Cs+(1/combs[:, 2]-Cs)/combs[:, 1]*(1/combs[:, 0]-1))
            calc = np.where(calc<=0, maxKf, calc) # replace negative Kf by Kf of pure water
            mini = max(minKf, np.nanmin(calc, axis = 0))
            maxi = max(minKf, np.nanmax(calc, axis = 0))
            return min(maxKf, value), mini, maxi

    def Get_Cf_from_B_reuss(s, B, n, K, Pg, Cw, heat_cap, Cf_model_B="Koning"):
        """"Returns acoustic fluid compressibility from B value (supposing incompressible grains) - reuss bound"""
        Sr = s.Get_Sr_from_B(B, K, n, 1/Pg, Cw, Cf_model_B)
        Sr = np.min((np.max((Sr, np.zeros(Sr.size, dtype = float)), axis = 0), np.ones(Sr.size, dtype = float)), axis = 0)
        return Sr*Cw+(1-Sr)*1/(heat_cap*Pg)

    def Get_Cf_from_B_voigt(s, B, n, K, Pg, Cw, heat_cap, Cf_model_B = "Koning"):
        """"Returns acoustic fluid compressibility from B value (supposing incompressible grains) - voigt bound"""
        Sr = s.Get_Sr_from_B(B, K, n, 1/Pg, Cw, Cf_model_B)
        Sr = np.min((np.max((Sr, np.zeros(Sr.size, dtype = float)), axis = 0), np.ones(Sr.size, dtype = float)), axis = 0)
        return 1/(Sr*1/Cw+(1-Sr)*heat_cap*Pg) #np.min((np.max((1/(Sr*1/Cw+(1-Sr)*1/Cg), np.ones(Sr.size, dtype = float)*Cw), axis = 0), np.ones(Sr.size, dtype = float)*Cg), axis = 0) #np.nanmin(np.nanmax((1/(Sr*1/Cw+(1-Sr)*1/Cg), Cw)), Cg)
        
    def Get_mass_layer(s, d, h):
        """Get mass of layer for dry compaction method"""
        e = s.emax-s.Id*(s.emax-s.emin)
        rd = s.rhos/(1+e)
        Vl = h*np.pi*d**2/4
        return rd*Vl
    
    def Get_Kf_from_Ku(s, n, Ku, K, Ks=-1, calculate_ci=False, Kuci=[0,0], nci=[0,0], Kci=[0,0], minKf=1e5, maxKf = 2.4e9):
        """
        Return the fluid bulk modulus from Ku. Ks=-1 if incompressible grains
        If calculate_ci = True : calculate resulting confidence interval from ciKu, cin and ciK
        For ci interval calculation : Ku must be a single value, not an array
        ci calculation only programmed with Ks different from 0
        """
        if Ks==-1:
            value= n*(Ku-K)
        else:
            value = (n*(Ku-K))/(1-(1+n)*K/Ks-(1-n)*Ku/Ks+K*Ku/(Ks**2))
        if calculate_ci==False:
            return value
        else:
            combs = np.zeros((2**3, 3), dtype = float)
            counter = 0
            for i, Kuval in enumerate(Kuci):
                for j, nval in enumerate(nci):
                    for k, Kval in enumerate(Kci):
                        combs[counter, 0] = Kuval
                        combs[counter, 1] = nval
                        combs[counter, 2] = Kval
                        counter+=1
            calc = (combs[:, 1]*(combs[:, 0]-combs[:, 2]))/(1-(1+combs[:, 1])*combs[:, 2]/Ks-(1-combs[:, 1])*combs[:, 0]/Ks+combs[:, 2]*combs[:, 0]/(Ks**2))  
            calc = np.where(calc<=0, minKf, calc)
            mini = max(minKf, np.nanmin(calc, axis = 0))
            maxi = max(value, minKf, np.nanmax(calc, axis = 0))
            return min(maxKf, value), mini, maxi



    def Get_Sr_from_Cf_Cw_Cg(s, Cf, Cw, Cg, maxSr = 1, minSr = 0, calculate_ci=False, Cf_ci = [0,0]):
        """
        Returns saturation ratio Sr from Cw, Cg and Cf
        Boundary conditions 0<=Sr<=1 by default
        If calculate_ci True : also returns confidence interval : only works when given a single Cf as input
        Supposes that the error comes exclusively from Cf
        Cf_ci is [Cfmin, Cfmax] : 
        """
        value = np.minimum(maxSr, np.maximum(minSr, (Cf-Cg)/(Cw-Cg)))
        if calculate_ci==False:
            return value
        else:
            mini = np.minimum(maxSr, np.maximum(minSr, (Cf_ci[1]-Cg)/(Cw-Cg)))
            maxi = np.minimum(maxSr, np.maximum(minSr, (Cf_ci[0]-Cg)/(Cw-Cg)))
            return value, mini, maxi

    def Get_Sr_from_Pg(s, Hcp, Pgsat, Pg, Mg, T):
        """Return saturation ratio estimated from Henry dissolution equilibrium"""
        R = 8.31446261815324  # Ideal g constant [J.K-1.mol-1]
        return np.minimum(1, 1/(1+(Hcp*(Pgsat-Pg)*Mg)/(Mg*Pg/(R*T))))
    
    #Error calculations
    def Calculate_error_Kf(s, n, Ku, K, dn, dKu, dK, Ks=-1, dKs = -1):
        """
        Uses differential calculations to estimate the error on Kf -  Ks = -1 indicates incompressible grains
        Supposes no error on grain bulk modulus
        All equations calculated using sympy, verified using wxMaxima
        """
        if Ks==-1:
            dkfdn = (Ku-K)
            dkfdku = n
            dkfdk = -n
        else:
            dkfdn = n*(-K + Ku)*(K/Ks - Ku/Ks)/(-K*(n + 1)/Ks + K*Ku/Ks**2 + 1 - Ku*(1 - n)/Ks)**2 + (-K + Ku)/(-K*(n + 1)/Ks + K*Ku/Ks**2 + 1 - Ku*(1 - n)/Ks)
            dkfdku = n*(-K + Ku)*(-K/Ks**2 + (1 - n)/Ks)/(-K*(n + 1)/Ks + K*Ku/Ks**2 + 1 - Ku*(1 - n)/Ks)**2 + n/(-K*(n + 1)/Ks + K*Ku/Ks**2 + 1 - Ku*(1 - n)/Ks)
            dkfdk = n*(-K + Ku)*((n + 1)/Ks - Ku/Ks**2)/(-K*(n + 1)/Ks + K*Ku/Ks**2 + 1 - Ku*(1 - n)/Ks)**2 - n/(-K*(n + 1)/Ks + K*Ku/Ks**2 + 1 - Ku*(1 - n)/Ks)
        return dkfdn*dn+dkfdku*dKu+dkfdk*dK
    
    def Calculate_error_Ku_plane_wave(s, dMu, dG):
        """Uses differential calculations to estimate the error on Ku""" 
        return dMu-4*dG/3

    def Calculate_error_modulus_from_velocity(s, vel, rho, dvel, drho):
        """Uses differential calculations to estimate the error on a modulus calculated from velocity""" 
        return vel**2*drho+2*rho*vel*dvel
    
    def Calculate_error_rho(s, n, rhow, Sr, drhod, dn, dSr):
        """
        Uses differential calculations to estimate the error on rho 
        Supposes no error on pure water density
        """ 
        return drhod+rhow*n*dSr+rhow*Sr*dn

    def Calculate_error_porosity(s, e, de):
        """Uses differential calculations to estimate the error on porosity"""
        return 1/((1+e)**2)*de

    def Calculate_error_dry_density(s, e, de):
        """Uses differential calculations to estimate the error on dry density rhod"""
        return -s.rhos/((1+e)**2)*de
    
    def Gassman_hill_p_modulus(s, Cw, Cg, G, Cs, Ck, n, Sr, 
                               calculate_ci=False, 
                               Cwci=[0,0],
                               Cgci=[0,0],
                               Gci=[0,0],
                               Ckci=[0,0],
                               nci=[0,0],
                               Srci=[0,0]
                                ):
        """
        Get Gassman-Hill P-wave modulus from Cw, Cg, G, Cs, Ck, n, Sr
        If calculate_ci, also returns confidence interval calculated based on
        ci from Cw, Cg, G, Ck, n, Sr : supposes no error on Cs
        ci must be provided as [min, max]
        When calculating ci : can only compute one value of Ku at a time
        """
        # M = K+4/3G
        Mgas = (1+n*Cg/Ck-(1+n)*Cs/Ck)/(n*Cg+(1-n)*Cs-Cs**2/Ck)+4/3*G
        Mwater = (1+n*Cw/Ck-(1+n)*Cs/Ck)/(n*Cw+(1-n)*Cs-Cs**2/Ck)+4/3*G
        value = 1/(Sr/Mwater+(1-Sr)/Mgas)
        if calculate_ci==False:
            return value
        else:
            #untested
            Cws = np.zeros(2**6, dtype = float)
            Cgs = np.zeros(2**6, dtype = float)
            Gs = np.zeros(2**6, dtype = float)
            Cks = np.zeros(2**6, dtype = float)
            ns = np.zeros(2**6, dtype = float)
            Srs = np.zeros(2**6, dtype = float)
            counter = 0
            for Cwval in Cwci:
                for Cgval in Cgci:
                    for Gval in Gci:
                        for Ckval in Ckci:
                            for nval in nci:
                                for Srval in Srci:
                                    Cws[counter]=Cwval
                                    Cgs[counter]=Cgval
                                    Gs[counter]=Gval
                                    Cks[counter]=Ckval
                                    ns[counter]=nval
                                    Srs[counter]=Srval
                                    counter+=1

            Mgass = (1+ns*Cgs/Cks-(1+ns)*Cs/Cks)/(ns*Cgs+(1-ns)*Cs-Cs**2/Cks)+4/3*Gs
            Mwaters = (1+ns*Cws/Cks-(1+ns)*Cs/Cks)/(ns*Cws+(1-ns)*Cs-Cs**2/Cks)+4/3*Gs
            values = 1/(Srs/Mwaters+(1-Srs)/Mgass)
            mini = np.nanmin(values, axis = 0)
            maxi = np.nanmax(values, axis = 0)
            return value, mini, maxi

# ---------------------------- Sand class---------------------------------------
class Sand(Soil):
    """Class for sands"""
    def __Init__(s, name):
        super().__init__(name)
        Soil.__Init__(s, name)

    G0A = 0. #GFP AIC paper
    G0B = 0. #GFP AIC paper
    G0C = 0. #GFP AIC paper
    G0D = 0. #parameter for fitting that does not depend on e/Id
    dG0D = 0. #error on parameter G0D : 0.95 interval normal law
    G0E = 0. #parameter for fitting that does not depend on e/Id
    dG0E = 0. #error on parameter G0E : 0.95 interval normal law
    Cu = 0. 
    D50 = 0.0
    Ks = 3.7e10 # Bulk modulus grains : Subramaniyanet al. 2015 (for Fontainebleau sandstone)
    # default : unloading bulk modulus (elastic)
    KA = 1.92 #parameter A for Hardin style fitting from drained consolidation
    dKA = 0.01 #error on A : 0.95 ci with Student law
    KB = 8.14 #parameter B for Hardin style fitting from drained consolidation
    dKB = 0.03 #error on B : 0.95 ci with Student law
    KC = 0.6645 #parameter C for Hardin style fitting from drained consolidation
    dKC = 0.0002 #error on C : 0.95 ci with Student law
    KD = 0.0 #parameter for fitting that does not depend on e/Id -> A
    dKD = 0.0 #error on D : 0.95 ci with student law
    KE = 0.0 #parameter for fitting that does not depend on e/Id -> C
    dKE = 0.0 #error on E : 0.95 ci with Student law
    dpois = 0.0 #error on poisson ratio
    Mcc = 0.0 #Critical state stress ratio in compression
    phics = -1.0 #Critical state friction angle 
    Drc0 = 0.0 #Parameter describing Critical state
    lc = 0.0 #Parameter describing Critical state
    xi = 0.7 #Parameter describing Critical state
    ec0_dm04 = 0.88 #Parameter describing Critical state
    lc_dm04 = 0.0717 #Parameter describing Critical state
    xi_dm04 = 0.236 #Parameter describing Critical state
    Cdr = -1.0 #Parameter Cdr for P2Psand small strain shear modulus
    g0 = -1.0 #Parameter g0 for P2Psand small strain shear modulus
    Cp2p = -1.0 #Parameter Cp2p for P2Psand small strain shear modulus
    h0p2p = -1.0  #Parameter h0 for P2Psand plastic hardening
    #modulus reduction
    gammar = -1.0 #parameter for stokoe type modulus reduction
    alpha_reduction = -1.0 #exponent for stokoe type modulus reduction

    def Update_hardin_K_params(s, phase = "Un"):
        """
        This function allows a choice between different parameter sets for K = f(p', e)\n
        - If phase = Un : Unloading : unloading (elastic) params\n
        - If phase = UL : Unconsolidated loading (elasto-plastic) params
        """
        if phase=="Un":
            s.KA = 1.92 #parameter A for Hardin style fitting from drained consolidation
            s.dKA = 0.01 #error on A : 0.95 ci with Student law
            s.KB = 8.14 #parameter B for Hardin style fitting from drained consolidation
            s.dKB = 0.03 #error on B : 0.95 ci with Student law
            s.KC = 0.6645 #parameter C for Hardin style fitting from drained consolidation
            s.dKC = 0.0002 #error on C : 0.95 ci with Student law
        elif phase=="UL":
            s.KA = 72.8 #parameter A for Hardin style fitting from drained consolidation
            s.dKA = 0.4 #error on A : 0.95 ci with Student law
            s.KB = 1.662 #parameter B for Hardin style fitting from drained consolidation
            s.dKB = 0.003 #error on B : 0.95 ci with Student law
            s.KC = 0.499 #parameter C for Hardin style fitting from drained consolidation
            s.dKC = 0.001 #error on C : 0.95 ci with Student law
        else:
            print("Please choose either Un or UL phases")

    def Get_hardin_drnevich_G0(s, pe, Pref = 101.325, calculate_ci=False, dpe = 0.0, de = 0.0):
        """
        Get and set G0 using Hardin & Drnevich (1972) equations : soil parameters must be previously given. 
        pe is effective stress in the same unit as Pref (default is kPa)
        Returned modulus is in MPA
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined before
        Supposes that highers error bound given by pe+dpe and lower by pe-dpe
        """
        # Pref usually taken as atmospheric pressure
        value = s.G0A*(s.G0B-s.e)**2/(1+s.e)*(pe/Pref)**s.G0C 
        if calculate_ci==False:
            return  value
        else:
            ci_g0a = [s.G0A-s.dG0A, s.G0A+s.dG0A]
            ci_g0b = [s.G0B-s.dG0B, s.G0B+s.dG0B]
            ci_g0c = [s.G0C-s.dG0C, s.G0C+s.dG0C]
            ci_e = [s.e-de, s.e+de]
            ci_pe = [pe-dpe, pe+dpe]
            g0as = np.zeros((2**5), dtype = float)
            g0bs = np.zeros((2**5), dtype = float)
            g0cs = np.zeros((2**5), dtype = float)
            pes = np.zeros((2**5), dtype = float)
            es = np.zeros((2**5), dtype = float)
            counter = 0
            for i, ga in enumerate(ci_g0a):
                for j, gb in enumerate(ci_g0b):
                    for k, gc in enumerate(ci_g0c):
                        for l, e in enumerate(ci_e):
                            for m, pe in enumerate(ci_pe):
                                g0as[counter]=ga
                                g0bs[counter]=gb
                                g0cs[counter]=gc
                                es[counter]=e
                                pes[counter]=pe
                                counter+=1
            values = g0as*(g0bs-es)**2/(1+es)*(pes/Pref)**g0cs 
            mini = np.nanmin(values, axis = 0)
            maxi = np.nanmax(values, axis = 0)
            return value, mini, maxi
        
    def Get_fitted_G0(s, pe, Pref = 101.325, calculate_ci = False, dpe = 0):
        """
        Get and set G0 using simplified Hardin & Drnevich (1972) equations that does not depend on void ratio: soil parameters must be previously given. 
        pe is effective stress in the same unit as Pref (default is kPa)
        Returned modulus is in MPA
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined before
        Supposes that highers error bound given by pe+dpe and lower by pe-dpe
        """
        #tested and validated with error calculation
        # Pref usually taken as atmospheric pressure
        value = s.G0D*(pe/Pref)**s.G0E 
        if calculate_ci==False:
            return  value
        else:
            cis = np.array([[s.G0D-s.dG0D, s.G0D+s.dG0D], [s.G0E-s.dG0E, s.G0E+s.dG0E]], dtype = float)
            combs = np.zeros((2**cis.shape[0], 2), dtype = float)
            counter = 0
            for i, ga in enumerate(cis[0, :]):
                for k, gc in enumerate(cis[1, :]):
                    combs[counter, :] = [ga, gc]
                    counter+=1
            params_ones = np.ones_like(combs[:, 0], dtype = float)
            maxi = np.nanmax(combs[:, 0]*np.outer(((pe+dpe)/Pref), params_ones)**combs[:, 1], axis = 1)
            mini = np.nanmin(combs[:, 0]*np.outer(((pe-dpe)/Pref), params_ones)**combs[:, 1], axis = 1)
            return np.array([value, mini, maxi], dtype = float)
        

    def Get_G0_menq(s, pe, Pref = 101.325):
        """
        Returns G0 estimated using Menq (2003) correlations in MPA\n
        -D50 : median grain diameter in mm
        """
        return 67.1*s.Cu**(-0.2)*s.e**(-1-(s.D50/20)**0.75)*(pe/Pref)**(0.48*s.Cu**0.09)
    



    def Get_hardin_drnevich_K(s, pe, Pref = 101.325, calculate_ci = False, dpe = 0):
        """
        Get and set K using Hardin & Drnevich (1972) equations : soil parameters must be previously given. 
        pe is effective stress in the same unit as Pref (default is kPa)
        Returned modulus is in MPA
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined before
        Supposes that highers error bound given by pe+dpe and lower by pe-dpe
        """
        #tested and validated with error calculation
        # Pref usually taken as atmospheric pressure
        value = s.KA*(s.KB-s.e)**2/(1+s.e)*(pe/Pref)**s.KC
        if calculate_ci==False:
            return  value
        else:
            cis = np.array([[s.KA-s.dKA, s.KA+s.dKA], [s.KB-s.dKB, s.KB+s.dKB], [s.KC-s.dKC, s.KC+s.dKC], [s.e-s.err_e, s.e+s.err_e]], dtype = float)
            combs = np.zeros((2**cis.shape[0], 4), dtype = float)
            counter = 0
            for i, ka in enumerate(cis[0, :]):
                for j, kb in enumerate(cis[1, :]):
                    for k, kc in enumerate(cis[2, :]):
                        for l, e in enumerate(cis[3, :]):
                            combs[counter, :] = [ka, kb, kc, e]
                            counter+=1
            params_ones = np.ones_like(combs[:, 0], dtype = float)
            maxi = np.nanmax(combs[:, 0]*(combs[:, 1]-combs[:, 3])**2/(1+combs[:, 3])*np.outer(((pe+dpe)/Pref), params_ones)**combs[:, 2], axis = 1)[0]
            mini = np.nanmin(combs[:, 0]*(combs[:, 1]-combs[:, 3])**2/(1+combs[:, 3])*np.outer(((pe-dpe)/Pref), params_ones)**combs[:, 2], axis = 1)[0]
            return np.array([value, mini, maxi], dtype = float)

    def Get_fitted_K(s, pe, Pref = 101.325, calculate_ci = False, dpe = 0):
        """
        Get and set K using simplified Hardin & Drnevich (1972) equations : soil parameters must be previously given. 
        pe is effective stress in the same unit as Pref (default is kPa)
        Returned modulus is in MPA
        If calculate_ci is True, returns value, min, max. Uses the ci on parameters defined before
        Supposes that highers error bound given by pe+dpe and lower by pe-dpe
        """
        # Pref usually taken as atmospheric pressure
        #tested and validated with error calculation
        value = s.KD*(pe/Pref)**s.KE 
        if calculate_ci==False:
            return  value
        else:
            cis = np.array([[s.KD-s.dKD, s.KD+s.dKD], [s.KE-s.dKE, s.KE+s.dKE]], dtype = float)
            combs = np.zeros((2**cis.shape[0], 2), dtype = float)
            counter = 0
            for i, ka in enumerate(cis[0, :]):
                for k, kc in enumerate(cis[1, :]):
                    combs[counter, :] = [ka, kc]
                    counter+=1
            params_ones = np.ones_like(combs[:, 0], dtype = float)
            maxi = np.nanmax(combs[:, 0]*np.outer(((pe+dpe)/Pref), params_ones)**combs[:, 1], axis = 1)[0]
            mini = np.nanmin(combs[:, 0]*np.outer(((pe-dpe)/Pref), params_ones)**combs[:, 1], axis = 1)[0]
            return np.array([value, mini, maxi], dtype = float)

     
    
    def Get_modulus_degradation_oztoprak(s, strain, pe, e, Id, Pref = 101.325):
        """
        Use works of Oztoprak & Bolton (2013) : strain can be an array, if so, returns a matrix
        All strains must be given in percentages (x100)
        """
        gammar = 0.01*s.Cu**(-0.3)*(pe/Pref)+0.08*e*Id
        gammae = 0.0002+0.012*gammar
        a=s.Cu**(-0.075)
        #res = np.ones(len(strain), len(pe), dtype = float)
        #res = res*strain
        return 1/(1+(np.maximum(0, ((strain-gammae)/gammar)))**a)
    
    def Get_modulus_degradation_menq(s, strain, pe, Pref = 101.325):
        """
        Use works of Menq (2003) : strain can be an array, if so, returns a matrix
        All strains must be given in percentages (x100)
        """
        gammar = 0.12*s.Cu**(-0.6)*(pe/Pref)**(0.5*s.Cu**(-0.15))
        a = 0.86+0.1*np.log10(pe/Pref) 
        return 1/(1+(np.maximum(0, ((strain)/gammar)))**a)
    
    def Get_modulus_reduction_stokoe(s, strain):
        """
        Get stokoe estimation of modulus reduction curve. Requires that gammar be previously defined in sand class\n
        Parameters:\n
        - strain: engineering strain = 2epsq in absolute value (no percent)
        """
        return 1/(1+(strain/s.gammar)**s.alpha_reduction)
    
    def Get_Wichtmann_damping_ratio(s, GG0):
        """
        Returns damping ratio from GG0 ratio based on works of Wichtmann and Triantafyllidis (2013) for quartz sand
        Requires uniformity coefficient to be declared
        """
        c1 = 0.26-0.074*np.log(s.Cu)
        c2 = -0.59+0.158*np.log(s.Cu)
        return c1*GG0**2+c2*GG0-(c1+c2)
    
    def Get_P2Psand_G0(s, Id, peff, Pref = 101.325):
        """
        Returns small strain shear modulus using expression from P2Psand soil model
        Needs parameters Cdr, g0 and Cp2p to be defined in corresponding sand class
        requires density index (not in percentage) and effective pressure, 
        By default, provide effective pressure in kPa and get modulus in MPa 
        """
        return s.g0*(Id+s.Cdr)*Pref*(peff/Pref)**s.Cp2p
    
    def Get_stress_ratio_from_friction(s, friction, compression = True, calculate_ci=False, ciM = [0,0]):
        """
        returns the stress ratio M corresponding to the given friction angle\n
        Parameters:\n
        - friction: angle, in degrees\n
        - compression : boolean: True if compression and False if extension\n
        """
        fa = np.radians(friction)
        if compression:
            return 6*np.sin(fa)/(3-np.sin(fa))
        else: #extension
            return 6*np.sin(fa)/(3+np.sin(fa))
    
    def Get_phi_from_M(s, M, compression = True, calculate_ci=False, ciM = [0,0]):
        """
        returns the fricion angle corresponding to the stress ratio\n
        Parameters:\n
        - stress ratio M\n
        - compression : boolean: True if compression and False if extension\n
        Returns:\n
        - friction: angle, in degrees\n
        - confidnence interval : mini and maxi
        """
        if compression:
            value = np.degrees(np.arcsin(3*M/(6+M)))
            if calculate_ci:
                Ms = np.array(ciM, dtype = float)
                phis = np.degrees(np.arcsin(3*Ms/(6+Ms)))    
                return value, np.nanmin(phis), np.nanmax(phis)
            else:
                return value
        else: #extension
            value = np.degrees(np.arcsin(3*M/(6-M)))
            if calculate_ci:
                Ms = np.array(ciM, dtype = float)
                phis = np.degrees(np.arcsin(3*Ms/(6-Ms)))    
                return value, np.nanmin(phis), np.nanmax(phis)
            else:
                return value

    def Get_critical_state_pc_p2p(s, Id, Patm = 101325):
        """
        returns the critical state effective stress p'c for a given density ratio\n
        Parameters:\n
        - Id : relative density \n
        - Patm : Atmospheric pressure. Unit of Patm defines unit of p'c
        """
        return Patm*((Id-s.Drc0)/s.lc)**(1/s.xi)
    
    def Get_critical_state_ec_dm04(s, pe, Patm = 101325):
        """
        returns the critical state effective stress p'c for a given density ratio\n
        Parameters:\n
        - pe : Effective stress (in the same unit as Patm) \n
        - Patm : Atmospheric pressure.
        """
        return s.ec0_dm04-s.lc_dm04*(pe/Patm)**s.xi_dm04
    
    def Get_Ir_from_Id_pe(s, Id, pe, Patm=101325):
        """
        Returns Bolton's state parameter Ir calculated from CS line\n
        Parameters:\n
        - Id : relative density \n
        - pe : relative effective stress (in the same unit as Patm)
        - Patm : Atmospheric pressure. Unit of Patm defines unit of p'c
        """
        pc = s.Get_critical_state_pc_p2p(Id, Patm)
        return Id*np.log(pc/pe)
    

# ---------------------------- Fontainebleau class---------------------------------------        

class NE34(Sand):
    """Class for Fontainebleau sand"""

    def __Init__(s, name):
        super().__init__(name)
        Sand.__Init__(s, name)

    emax = 0.87  # Max1void index from Andria-Ntoanina (2011)
    emin = 0.54  # Min void index from Andria-Ntoanina (2011)
    rhos = 2.65e3  # specific mass of the grains [kg/m3]
    poisson = 0.18  # 
    poisson_noId = 0.18
    rhod = 1.68e3  # Andria-Ntoanina (2011)
    Ks = 3.7e10 # Bulk modulus grains : Subramaniyanet al. 2015 (for Fontainebleau sandstone)
    data_origin = "Flood-Page et al. (2022)"  # origin of the data
    D50 = 0.223 #Andria-Ntoanina (2011)
    G0A = 132.98 #saturated be tests
    dG0A = 4.27 #saturated be tests
    G0B = 1.93 #saturated be tests
    dG0B = 0.021 #saturated be tests
    G0C = 0.44 #saturated be tests
    dG0C = 0.02 #saturated be tests
    G0D = 99.98 #parameter for fitting that does not depend on e/Id
    dG0D = 0.26 #error on parameter G0D : 0.95 interval normal law
    G0E = 0.52 #parameter for fitting that does not depend on e/Id
    dG0E = 0.005 #error on parameter G0E : 0.95 interval normal law
    Cu = 1.5 #andria
    KA = 19.99 #parameter A for Hardin style fitting from drained consolidation
    dKA = 5.03 #error on A : 0.95 ci with Student law
    KB = 2.98 #parameter B for Hardin style fitting from drained consolidation
    dKB = 0.3 #error on B : 0.95 ci with Student law
    KC = 0.46 #parameter C for Hardin style fitting from drained consolidation
    dKC = 0.011 #error on C : 0.95 ci with Student law
    KD = 0.0 #parameter for fitting that does not depend on e/Id -> A
    dKD = 0.0 #error on D : 0.95 ci with student law
    KE = 0.0 #parameter for fitting that does not depend on e/Id -> C
    dKE = 0.0 #error on E : 0.95 ci with Student law
    dpois = 0.001 # from paper on aic
    Mcc = 1.293 #Critical state stress ratio in compression from my analysis of Benahmed/Azeiteiro papers
    phics = 30.0 #Critical state friction angle from phipeak/psipeak analysis
    #Critical state parameters
    Drc0 = 0.0 #Parameter describing Critical state
    lc = 0.2208 #Parameter describing Critical state
    xi = 0.1582 #Parameter describing Critical state
    ec0_dm04 = 0.88 #Parameter describing Critical state #Hussein Bakri
    lc_dm04 = 0.0717 #Parameter describing Critical state #Hussein Bakri
    xi_dm04 = 0.236 #Parameter describing Critical state #Hussein Bakri
    #Small strain elasticity
    Cdr = 0.68 #Parameter Cdr for P2Psand small strain shear modulus
    g0 = 970 #0.97 #Parameter g0 for P2Psand small strain shear modulus
    Cp2p = 0.44 #Parameter Cp2p for P2Psand small strain shear modulus
    #Bounding and pts
    h0p2p = 0.609 #0.62  #Parameter h0 for P2Psand plastic hardening
    nd = 0.0657 #0.502 # undrained #0.0657 # drained # 0.0152 # parameter controlling position of pts
    nb = 0.0843 #0.077 #0.0925 # parameter controlling position of peak


    def Reset(s):
        s.Id = 10000.0  # default density index
        s.emax = 0.87  # Max1void index
        s.emin = 0.54  # Min void index
        s.e0 = 0.0
        s.n0 = 0.0
        s.rhos = 2.65e3  # specific mass of the grains [kg/m3]
        s.rhod = 1.68e3  # dry mass density of the grains [kg/m3] : hypothesis that density does not change
        s.poisson = 0.18  # poisson coefficient from GFP
        s.poisson_noId = 0.18 # poisson coefficient from GFP - not sensitive to density

    def Get_phip(s, peff, Id):
        return 29.6+5.08*(Id*(5.6+0.58*np.log(peff)-np.log(peff))-0.62)
    
    def Get_li2013_degradation(s, peff, strain):
        """
        Returns the G/G0 ratio calculated by Li et al., 2013 correlation for fontainebleau sand NE34.
        Requires confining pressure peff in kPa and engineering shear strain in absolute value (no %) = 2sqrt(J2) = 2epsq in triaxial conditions
        """
        #To be tested
        if strain==0.:
            return 1.0
        else:
            K = 0.5*(1+np.tanh(np.log((0.000102/strain)**0.613)))
            mdif = 0.34*(1-np.tanh(np.log((0.000556/strain)**0.4)))
            GG0 = K*peff**mdif
            return GG0
        
    def get_li2013_damping(s, GG0):
        """
        Returns the damping percentage (x100) calculated by Li et al., 2013 correlation for fontainebleau sand NE34
        Requires shear modulus degradation GG0
        """
        #To be tested
        return 25.3*(0.513*GG0**2-1.351*GG0+1)


    def Reset_Hardin_Drnevich_params(s, datasource="Flood-Page et al. (2022)"):
        #default
        s.KA = 0.
        s.dKA = 0.
        s.KB = 0.
        s.dKB = 0.
        s.KC = 0.
        s.dKC = 0.
        s.KD = 0. #for overconsolidated unloading - no impact of void ratio (so)
        s.dKD = 0.
        s.KE = 0.
        s.dKE = 0.
        s.data_origin = datasource
        if datasource=="Andria-Ntoanina (2011)":
            s.G0A = 63.7 #Hussein - andria
            s.G0B = 2.17 #Hussein - andria
            s.G0C = 0.43 #Hussein - andria
            s.poisson = 0.2  #me
            s.G0D = 0.0
            s.G0E = 0.0
        elif datasource=="Delfosse-Ribay et al. (2004)":
            s.G0A = 68.2 #Hussein - delfosse
            s.G0B = 2.17 #Hussein - delfosse
            s.G0C = 0.47 #Hussein - delfosse
            s.poisson = 0.2  # me
            s.G0D = 0.0
            s.G0E = 0.0
            s.poisson_noId = 0.2
        elif datasource=="Flood-Page et al. (2022)":
            s.G0A = 72.7 #GFP AIC paper
            s.G0B = 2.17 #GFP AIC paper
            s.G0C = 0.49 #GFP AIC paper
            s.poisson = 0.18  # me
            s.G0D = 99.98
            s.dG0D = 0.26 #error on parameter G0D : 0.95 interval normal law
            s.G0E = 0.52 #parameter for fitting that does not depend on e/Id
            s.dG0E = 0.005 #error on parameter G0E : 0.95 interval normal law
            s.poisson_noId = 0.18
            s.KA = 19.99 #parameter A for Hardin style fitting from drained consolidation
            s.dKA = 5.03 #error on A : 0.95 ci with Student law
            s.KB = 2.98 #parameter B for Hardin style fitting from drained consolidation
            s.dKB = 0.3 #error on B : 0.95 ci with Student law
            s.KC = 0.46 #parameter C for Hardin style fitting from drained consolidation
            s.dKC = 0.011 #error on C : 0.95 ci with Student law
        elif datasource=="Flood-Page et al. (2023) - UC":
            s.G0A = 72.7 #GFP AIC paper
            s.G0B = 2.17 #GFP AIC paper
            s.G0C = 0.49 #GFP AIC paper
            s.poisson = 0.18  # me
            s.G0D = 99.98
            s.dG0D = 0.26 #error on parameter G0D : 0.95 interval normal law
            s.G0E = 0.52 #parameter for fitting that does not depend on e/Id
            s.dG0E = 0.005 #error on parameter G0E : 0.95 interval normal law
            s.poisson_noId = 0.18
            s.KA = 22.12 #parameter A for Hardin style fitting from drained consolidation
            s.dKA = 3.63 #error on A : 0.95 ci with Student law
            s.KB = 2.62 #parameter B for Hardin style fitting from drained consolidation
            s.dKB = 0.16 #error on B : 0.95 ci with Student law
            s.KC = 0.6 #parameter C for Hardin style fitting from drained consolidation
            s.dKC = 0.011 #error on C : 0.95 ci with Student law
        elif datasource=="Flood-Page et al. (2023) - OC - Loading":
            s.G0A = 72.7 #GFP AIC paper
            s.G0B = 2.17 #GFP AIC paper
            s.G0C = 0.49 #GFP AIC paper
            s.poisson = 0.18  # me
            s.G0D = 99.98
            s.dG0D = 0.26 #error on parameter G0D : 0.95 interval normal law
            s.G0E = 0.52 #parameter for fitting that does not depend on e/Id
            s.dG0E = 0.005 #error on parameter G0E : 0.95 interval normal law
            s.poisson_noId = 0.18
            s.KA = 19.99 #parameter A for Hardin style fitting from drained consolidation
            s.dKA = 5.03 #error on A : 0.95 ci with Student law
            s.KB = 2.98 #parameter B for Hardin style fitting from drained consolidation
            s.dKB = 0.3 #error on B : 0.95 ci with Student law
            s.KC = 0.46 #parameter C for Hardin style fitting from drained consolidation
            s.dKC = 0.011 #error on C : 0.95 ci with Student law
        elif datasource=="Flood-Page et al. (2023) - OC - Unloading":
            s.G0A = 72.7 #GFP AIC paper
            s.G0B = 2.17 #GFP AIC paper
            s.G0C = 0.49 #GFP AIC paper
            s.poisson = 0.18  # me
            s.G0D = 99.98
            s.dG0D = 0.26 #error on parameter G0D : 0.95 interval normal law
            s.G0E = 0.52 #parameter for fitting that does not depend on e/Id
            s.dG0E = 0.005 #error on parameter G0E : 0.95 interval normal law
            s.poisson_noId = 0.18
            s.KD = 72.78 #parameter A/D for simplified Hardin style fitting from drained consolidation
            s.dKD = 0.15 #error on D : 0.95 ci with Student law
            s.KE = 0.69 #parameter C/E for simplified Hardin style fitting from drained consolidation
            s.dKE = 0.002 #error on E : 0.95 ci with Student law
        elif datasource=="BE - Manual - Selected - Grade1":
            s.G0A = 68.85 #me
            s.G0B = 2.17 #me
            s.G0C = 0.49 #me
            s.poisson = 0.16
            s.G0D = 94.55
            s.G0E = 0.53
            s.poisson_noId = 0.16
        elif datasource=="BE - Manual - All - Grade1":
            s.G0A = 70.47 #me
            s.G0B = 2.17 #me
            s.G0C = 0.46 #me
            s.poisson = 0.18
            s.G0D = 98.55
            s.G0E = 0.48
            s.poisson_noId = 0.18
        elif datasource=="BE - Mixed - Selected - Grade1":
            s.G0A = 72.7 #me
            s.G0B = 2.17 #me
            s.G0C = 0.49 #me
            s.poisson = 0.18
            s.G0D = 99.98
            s.G0E = 0.52
            s.poisson_noId = 0.18
        elif datasource=="BE - Mixed - All - Grade1":
            s.G0A = 73.46 #me
            s.G0B = 2.17 #me
            s.G0C = 0.47 #me
            s.poisson = 0.19
            s.G0D = 102.78
            s.G0E = 0.49
            s.poisson_noId = 0.19
        elif datasource=="BE - SLA - Selected - Grade1":
            s.G0A = 72.7 #me
            s.G0B = 2.17 #me
            s.G0C = 0.49 #me
            s.poisson = 0.2
            s.G0D = 99.98
            s.G0E = 0.52
            s.poisson_noId = 0.2
        elif datasource=="BE - Manual - Selected - Grade3":
            s.G0A = 69.40 #me
            s.G0B = 2.17 #me
            s.G0C = 0.49 #me
            s.poisson = 0.17
            s.G0D = 94.79
            s.G0E = 0.53
            s.poisson_noId = 0.17
        elif datasource=="BE - Mixed - Selected - Grade3":
            s.G0A = 72.71 #me
            s.G0B = 2.17 #me
            s.G0C = 0.48 #me
            s.poisson = 0.18
            s.G0D = 99.52
            s.G0E = 0.51
            s.poisson_noId = 0.17
        elif datasource=="BE - Mixed - All - Grade3":
            s.G0A = 71.83 #me
            s.G0B = 2.17 #me
            s.G0C = 0.46 #me
            s.poisson = 0.19
            s.G0D = 100.96
            s.G0E = 0.47
            s.poisson_noId = 0.19
        elif datasource=="BE - SLA - Selected - Grade3":
            s.G0A = 72.71 #me
            s.G0B = 2.17 #me
            s.G0C = 0.48 #me
            s.poisson = 0.18
            s.G0D = 99.52
            s.G0E = 0.51
            s.poisson_noId = 0.18
        elif datasource=="Drained consolidation - Unconsolidated":
            s.KA = 22.12 #me
            s.KB = 2.62 #me
            s.KC = 0.6 #me
        elif datasource=="Drained consolidation - Consolidated, Unloading":
            s.KA = 6.29 #me
            s.KB = 5 #me
            s.KC = 0.63 #me
        elif datasource=="Drained consolidation - Consolidated, Reloading":
            s.KA = 19.99 #me
            s.KB = 2.98 #me
            s.KC = 0.46 #me
        # elif datasource=="Experiments - SLA/Unconsolidated":
        #     s.G0A = 66.63 #me
        #     s.G0B = 2.17 #me
        #     s.G0C = 0.46 #me
        # elif datasource=="Experiments - Manual - Consolidated":
        #     s.G0A = 63.66 #me
        #     s.G0B = 2.17 #me
        #     s.G0C = 0.4 #me
        # elif datasource=="Experiments - SLA/Consolidated":
        #     s.G0A = 66.52 #me
        #     s.G0B = 2.17 #me
        #     s.G0C = 0.41 #me
        else:
            print("The selcted datasource is unknown")

    def Get_G0(s, pe):
        """Get the small strain shear modulus (in Pa) from effective stress (in Pa) - deprecated : use """
        # Hussein Bakri with Andria-Ntoanina (2011) data
        s.G0 = (20.11/(0.5*s.e0**2 + 0.019))*(pe/101300)**0.43*1e6  # set
        return s.G0
        # Equation uses relative pressures in kPa, results in Pa

# ---------------------------- Toyoura class---------------------------------------
class Toyoura(Sand):
    """Class for Toyoura sand"""

    def __Init__(s, name):
        super().__init__(name)
        Sand.__Init__(s, name)

    emax = 0.975  # Max1void index from Tsukamoto et al. (2002)
    emin = 0.616  # Min void index from Tsukamoto et al. (2002)
    rhos = 2.65e3  # specific mass of the grains [kg/m3]
    poisson = 0.25  # poisson coefficient fro Li et al. (2013) : wrong
    Drc0 = -0.1995 #Parameter describing Critical state 
    lc = 0.0735 #Parameter describing Critical state
    xi = 0.6157 #Parameter describing Critical state
    phics = 31.6 #Critical state friction angle from Cheng & Detournay (2021)
    nb = 0.077
    D50 = 0.185 #kido et al. (2020) 0.216 #gu et al. (2013)
    Cu = 1.6 #Kido et al. (2020)
    G0A = 95.5 #kPa, from Gu et al. (2013)
    G0B = 2.17 #Gu et al. (2013)
    G0C = 0.41 # Gu et al. (2013)
    # def Reset(s):
    #     s.Id = 10000.0  # default density index
    #     s.emax = 0.975  # Max1void index #Kido et al. (2020)
    #     s.emin = 0.614  # Min void index #Kido et al. (2020)
    #     s.e0 = 0.0
    #     s.n0 = 0.0
    #     s.rhos = 2.65e3  # specific mass of the grains [kg/m3]
    #     s.rhod = 1.68e3  # dry mass density of the grains [kg/m3] : hypothesis that density does not change
    #     s.poisson = 0.25  # poisson coefficient fro Li et al. (2013) : questionnable

class Karlsruhe_fine_sand(Sand):
    """Class for Karlsruhe sand. Data all taker from Wichtmann et Triantafyllidis (2016) unless specified otherwise"""

    def __Init__(s, name):
        super().__init__(name)
        Sand.__Init__(s, name)

    # small strain parameters from Wichtmann, T., Fuentes, W. & Triantafyllidis, T. Inspection of three sophisticated constitutive models based on monotonic and cyclic tests on fine sand: Hypoplasticity vs. Sanisand vs. ISA. Soil Dynamics and Earthquake Engineering 124, 172–183 (2019).
    G0A = 150.0
    G0B = 2.97
    G0C = 0.5 
    emax = 1.054  # Max1void index 
    emin = 0.677  # Min void index 
    rhos = 2.65e3  # specific mass of the grains [kg/m3]
    poisson = 0.22  # poisson coefficient 
    Drc0 = -0.13#-0.1995 #Parameter describing Critical state 
    lc = 0.32#0.0735 #Parameter describing Critical state
    xi = 0.205 #0.6157 #Parameter describing Critical state
    nb = 0.1 #0.077
    nd = 0.04
    phics = 32.9 #31.3 #31.4 #30.6 #31.4 #32.68 #Critical state friction angle from same analysis
    ec0_dm04 = 1.10 # Wichtmann et al. (2019) 1.0 #1.103 #0.99 #Parameter describing Critical state
    lc_dm04 = 0.12 # Wichtmann et al. (2019) 0.05 #0.122 #0.05#0.042 #Parameter describing Critical state
    xi_dm04 = 0.21 # Wichtmann et al. (2019) 0.5#0.205 #0.5 #Parameter describing Critical state
    g0_dm04 = 150 #dm04 g0 parameter : fitted from Azeiteiro (2017) data 
    nb_dm04 = 1.58 #drained 1.1 # parameter position bounding surface dm04 :
    nb_dm04_undrained = 1.1 # parameter position bounding surface dm04 :
    nb_dm04_drained = 1.58 # parameter position bounding surface dm04 :
    nd_dm04 = 0.73 # drained parameter position pts surface dm04
    nd_dm04_undrained = 0.8 # undrained parameter position pts surface dm04
    nd_dm04_drained = 0.73 #drained
    Ad0_dm04 =  1.0065 
    h0_dm04 = 1.535
    ch_dm04 =  0.8013

    Cdr = 1.33 #Parameter Cdr for P2Psand small strain shear modulus
    g0 = 202 #Parameter g0 for P2Psand small strain shear modulus
    Cp2p = 0.5 #Parameter Cp2p for P2Psand small strain shear modulus

# ---------------------------- Hostun_rf class---------------------------------------
class HostunRF(Sand):
    """Class for Hostun RF sand"""

    def __Init__(s, name):
        super().__init__(name)
        Sand.__Init__(s, name)

    emax = 1.0  # Max1void index from Benahmed (2001)
    emin = 0.656  # Min void index from Benahmed (2001)
    rhos = 2.65e3  # specific mass of the grains [kg/m3]
    poisson = 0.196 #from correlation used in P2Psand  0.18  # from Azeiteiro et al. 2017
    Ks = 3.7e10 # Bulk modulus grains : Subramaniyanet al. 2015 (for Fontainebleau sandstone)
    Cu = 1.57 #Benahmed 2001
    G0A = 21.46 #data afrom Azeiteiro, GFP fit
    dG0A = 0.67 #data afrom Azeiteiro, GFP fit
    G0B = 3.39 #data afrom Azeiteiro, GFP fit
    dG0B = 0.042 #data afrom Azeiteiro, GFP fit
    G0C = 0.49 #data afrom Azeiteiro, GFP fit
    dG0C = 0.04 #data afrom Azeiteiro, GFP fit
    KD = 23.5 #from consolidation of Tran et al.
    dKD = 0.9 #from consolidation of Tran et al.
    KE = 0.28 #from cons Tran et al.
    dKE = 0.08 #from consolidation of Tran et al.
    g0_dm04 = 296 #dm04 g0 parameter : fitted from Azeiteiro (2017) data 
    # G0D = 99.98 #parameter for fitting that does not depend on e/Id
    # dG0D = 0.26 #error on parameter G0D : 0.95 interval normal law
    # G0E = 0.52 #parameter for fitting that does not depend on e/Id
    # dG0E = 0.005 #error on parameter G0E : 0.95 interval normal law
    #Mcc = 1.27 #Critical state stress ratio in compression from my analysis of Benahmed/Azeiteiro papers
    phics = 31.4 #31.3 #31.4 #30.6 #31.4 #32.68 #Critical state friction angle from same analysis
    ec0_dm04 = 1.04 # 1.0 #0.93 # drained 0.97 #Parameter describing Critical state
    lc_dm04 = 0.11 #0.07 #0.007 #drained 0.04#0.042 #Parameter describing Critical state
    xi_dm04 = 0.25 #0.36 #1.2 #drained 0.5 #Parameter describing Critical state
    Drc0 = -0.1 #0.08 #0.077 #0.0 #0.0825 #0.0 #Parameter describing Critical state 
    lc = 0.3 #0.12 #0.122 #0.1132 #0.1983 #0.1217#0.237 #Parameter describing Critical state
    xi = 0.25 #0.5 #0.519 #0.7 #0.3388 #0.4663 #0.32 #Parameter describing Critical state
    Cdr = 1.40 #Parameter Cdr for P2Psand small strain shear modulus
    g0 = 399 #Parameter g0 for P2Psand small strain shear modulus
    Cp2p = 0.49 #Parameter Cp2p for P2Psand small strain shear modulus
    h0p2p = 3.0 #0.2113  #Parameter h0 for P2Psand plastic hardening
    nb = 0.12 # 0.16 #0.10 #0.11 #0.10
    nb_drained = 0.09
    nb_undrained = 0.12
    nd = 0.08 #drained #undrained : 0.09
    nd_drained = 0.01
    nd_undrained = 0.08 
    nb_dm04 = 1.73 #1.59 # parameter position bounding surface dm04 :
    nb_dm04_undrained = 1.73 #1.6 # parameter position bounding surface dm04 :
    nb_dm04_drained = 4.3 ##!!!!!## only GFP data #1.59 # parameter position bounding surface dm04 :
    nd_dm04 = 1.3 # 0.19 #0.19 # undrained parameter position pts surface dm04
    nd_dm04_undrained = 1.3 #1.7 # undrained parameter position pts surface dm04
    nd_dm04_drained = 0.19 #0.19 #drained
    alpha_reduction = 0.996 #goudarzy et al. (2018) : fines content=0
    gammar100 = 1.1e-3 #1.48e-3 #from goudarzy et al. (2018)
    gammar = 1.1e-3  #from goudarzy et al. (2018) for p'=100kPa

    def Set_gammar_Hostun(s, peff, Pref=101.325):
        """Sets and returs modulus reduction parameter gamma_r for Stokoe reduction curve. Uses relation from Goudarzy et al. (2018)"""
        s.gammar = s.gammar100*(peff/Pref)**0.447
        return s.gammar
