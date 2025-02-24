# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:29:03 2024
BE-ROCKET Dedicated HRE Prelim design tool.
This script is created as a preliminary design tool, enabling the team
to design HRE propulsion systems, without having to wait for the BE-FORGE code to be
on point.
@author: Bram Samyn

Tank design contribution by Niels Baele
"""

#%% IMPORTING PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sympy as sy
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.integrate import simpson
import CEA_Wrap as CW
from CEA_Wrap import Fuel, Oxidizer, RocketProblem, ThermoInterface, DataCollector #https://github.com/civilwargeeky/CEA_Wrap
import CoolProp                                   # fluid properties http://www.coolprop.org
from CoolProp.CoolProp import PropsSI             # to calulate properties of substances
from CoolProp.Plots import PropertyPlot           # to plot property diagrams
from CoolProp.Plots import SimpleCompressionCycle # to plot cycles on diagrams
from sympy import Symbol, latex
import gmsh
import ezdxf

#%% ENGINE PROFILES
HRE_1 = {
        "name":'BigFlameyCandle',       #BFC
        "EngineType":"HRE",     #Indicates engine type: SRM ; HRE ; LRE ; PDE ; RDE
        "Fuel": "C32H66",         #Use the notations, as used in the CEARUN software use CEA_Wrap.open_thermo_lib() to view all species
        "Fuel_name": "paraffin", 
        "Oxidizer": "N2O",    #Also specify if the component is Gas (G) Liquid (L) or Solid (S)
        "T_f": 298.15,           #Fuel temperature [K]
        "T_o": 300,              #Oxidizer temperature [K]
        "DesignThrust": 1000,     #design thrust in [N]
        "DesignChamberPressure": 20, #Design chamber pressure in [Bar]
        "Popt": 101325 ,        #Pressure at optimum expansion
        "Feed":"PF",            #Feed system type: PF 'pressure fed', TP 'Turbo Pump'
        "OF": 7,             #Design Oxidizer-fuel ratio
        "tb": 1,            #preset for the burn time
        "Rho_f": 900,           #density of the fuel grain [kg/m³]
        "m_tank": 1,
        "m_fuel": 1,            #mass of the fuel '1' is a placeholder [kg]
        "m_o_dot": 1,           #mass flow rate of the oxidizer (initial) [kg/s]
        "a0": 15.5e-2,          #regression rate coefficient
        "k": 0.5,               #regression rate exponent
        "InjectorType": "Impinging-Unlike",  #Injector type
        "DPi": 0.2,             #injector pressure drop, as a fraction of Pc
        "Cdi": 0.8 ,             #discharge coefficient, look at table on injector orifice design
        "n_inj":4,              #number of injector elements
        "SF_w":1.1,             #Wall normal operation safety factor
        "SF_w_burst": 2,        #burst safety factor
        "Sigma_y": 70e6,        #Yield strength Copper 70e6 / steel 235e6-275e6
        "Sigma_UTS": 220e6,     #UTS copper 220e6 / steel 360e6-410e6
        "Di_rocket": 130e-3,    #inner diameter of the rocket skin
        "tank_material": "metal",       #tank material, either metal or composite
        "wall thickness" : 10     #temporary defined wall thickness in mm
    }

#%% CEA: Chemical Equilibrium Analysis

def CEA_RUN(Profile):

    fuel = Fuel(Profile["Fuel"], temp=Profile["T_f"]) 
    ox = Oxidizer(Profile["Oxidizer"], temp=Profile["T_o"])
    
    if Profile["EngineType"] == "SRM" or Profile["EngineType"] == "HRE" or Profile["EngineType"] =="LRE":
        Case = RocketProblem(pressure = Profile["DesignChamberPressure"],pressure_units="bar", materials=[fuel, ox], o_f=Profile["OF"],fac_ac = 6, pip=Profile["DesignChamberPressure"]/(Profile["Popt"]/100000)) 
        results = Case.run()
        #Isentropic ratio of specific heats
        gammas_t = results.t_gammas
        gammas_c = results.c_gammas 
        #Real ratio of specific heats
        gamma_t   = results.t_gamma
        gamma_c   = results.c_gamma
        Tc          = results.c_t
        Tt          = results.t_t
        Pe          = results.p
        Pc          = results.c_p 
        Cp_e        = results.cp * 1e3
        Cp_c        = results.c_cp * 1e3
        Rho_c       = results.c_rho
        Rho_t       = results.t_rho
        a_c         = results.c_son
        a_t         = results.t_son
        Cf          = results.cf
        Cstar       = results.cstar
        Mach_e      = results.mach
        ISP_e       = results.isp
        ISP_t       = results.t_isp
        ISP_vac     = results.ivac
        R_spec      = (1/Rho_c)*(Pc*10**5)/Tc
        Pr_c        = results.c_pran        #Prandtl number of combustion gasses
        K_c         = results.c_cond        #Thermal conductivity of combustion gasses W/mK at chamber
        K_t         = results.t_cond        #Thermal conductivity of combustion gasses W/mK at throat
        visc_t      = results.t_visc        #throat viscosity [Pa s]
        visc_c      = results.c_visc        #chamber viscosity [Pa s]
        #general parameters
        Rho_Ox_g =  PropsSI('D','T',Profile["T_f"],'P',Pc*1e5,"O2")
        Rho_Ox_L = PropsSI('D','T',Profile["T_o"],'P',Pc*1e5,"O2")
        I_o = PropsSI('I','P',Pc*1e5,'Q',0, "O2")   #Surface Tension N/m or Dynes/cm   
        µo = PropsSI('V','T',Profile["T_f"],'P',Pc*1e5,"O2") #Viscosity of oxidizer Pa s
        Ko = PropsSI('L','T',Profile["T_f"],'P',Pc*1e5,"O2") #Thermal Conductivity of oxidizer W/mK         
        
        return {"Tc":Tc,"Tt":Tt,"Pc":Pc,"Pe":Pe, "Cp_c":Cp_c,"Cp_e":Cp_e,"Rho_c":Rho_c,"Rho_t":Rho_t,
                "RhooG":Rho_Ox_g,"RhooL":Rho_Ox_L,"a_c":a_c,"a_t":a_t,
                "Cf":Cf, "C*":Cstar,"ISP_e":ISP_e,"ISP_t":ISP_t,"ISP_vac":ISP_vac,"gamma_c":gamma_c,"gamma_t":gamma_t,
                "Rspec":R_spec,"Me":Mach_e,"Kc":K_c,"Kt":K_t, "Pr":Pr_c,"I_o":I_o,"µo":µo,"Ko":Ko,"visc_t":visc_t,
                "visc_c":visc_c}

#%% O/F RATIO ANALYSIS
def OF_ANSYS(Profile):
    fuel = Fuel(Profile["Fuel"], temp=Profile["T_f"]) 
    ox = Oxidizer(Profile["Oxidizer"], temp=Profile["T_o"])
    OF_Range = np.arange(3.5,12,0.1)
    # Rocket at 30 Bar chamber pressure and an expansion ratio of 220
    if Profile["EngineType"] == "SRM" or Profile["EngineType"] == "HRE" or Profile["EngineType"] =="LRE":
        ISP_e_list =[]
        ISP_vac_list =[]
        C_star_list =[]
        Cf_list = []
        for i in range(len(OF_Range)):
            Case = RocketProblem(pressure = Profile["DesignChamberPressure"],pressure_units="bar", materials=[fuel, ox], o_f=OF_Range[i],fac_ac = 6, pip=Profile["DesignChamberPressure"]/(Profile["Popt"]/100000)) 
            results = Case.run()
            ISP_e_list.append(results.isp)
            ISP_vac_list.append(results.ivac)
            C_star_list.append(results.cstar)
            Cf_list.append(results.cf)
                
        fig, ax1 = plt.subplots()
        plt.grid()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.2))
        ax1.plot(OF_Range,ISP_e_list,color='blue',label="Isp [s]")
        ax2.plot(OF_Range,C_star_list,color='red',label="C* [m/s]")
        ax3.plot(OF_Range,Cf_list,color='green',label="Cf[/]")
        ax1.set_xlabel("O/F ratio")
        ax1.set_xticks(np.arange(3,13,1))
        ax1.set_ylabel("Specific impulse [s]")
        ax2.set_ylabel("Characteristic Vel. [m/s]", color ='red')
        ax3.set_ylabel("Thrust Coeff. [/]",color ='green')
        ax2.tick_params(axis='y', colors='red')
        ax3.tick_params(axis='y', colors='green')
        ax2.spines['right'].set_color('red')
        ax3.spines['right'].set_color('green')
        fig.legend(loc="lower center", bbox_to_anchor=(0.5,0), bbox_transform=ax1.transAxes)
        fig.suptitle(f"Efficiency factors vs O/F ({Profile['Fuel']} & {Profile['Oxidizer']})", fontsize=12)
        plt.tight_layout()
        plt.show()
        
        print("Optimum O/F for Isp:",OF_Range[np.argmax(ISP_e_list)] )
        print("Optimum O/F for C*:",OF_Range[np.argmax(C_star_list)] )
        print("Optimum O/F for Cf:",OF_Range[np.argmax(Cf_list)] )
        
        
    return {"C*":C_star_list,"Cf":Cf_list,"Isp":ISP_e_list,"OF":OF_Range}

#OF_ANSYS(HRE_1)

def It_Req(Profile,m_rocket,Launch_angle,TgtAlt,Perf_eff):
    #First lets solve the minimum required average thrust to leave the rail with a velocity of 30 m/s
    v_rail_min = 35      #minimum allowable exit velocity of the rail
    l_rail = 12         #launch rail length
    g0 = 9.80665
    def T_avg(Tavg):
        return (1/2)*(Tavg/m_rocket - g0*np.sin(Launch_angle*np.pi/180) )*(v_rail_min/(Tavg/m_rocket - g0*np.sin(Launch_angle*np.pi/180) ))**2 - l_rail
    Avg_Thrust = float(fsolve(T_avg,1000)[0])  #the 1000 is an initial value used for solving the equation
    OF_prof = OF_ANSYS(Profile)
    Rho_Ox_L = PropsSI('D','T',Profile["T_o"],'P',80*1e5,Profile["Oxidizer"]) #only applicable for pressure fed, note to change the tank pressure
    m_dot_list =[]
    tb_list =[]
    m_prop_list  =[]
    m_ox_list =[]
    m_fuel_list = []
    Alt_list =[]
    I_tot_list =[]
    V_ox =[]
    M_tank =[]
    L_tank =[]
    M_system =[]
    #%% Oxidizer tank mass
    Pmax = 8*1e6            #Pa
    Pmin = 3*1e6            #Pa
    Mass_f = 0.603          #kg
    Dtank = Profile["Di_rocket"]           #m
    SF_metallic = 2         #EuRoC
    SF_composite = 3        #EuRoC
    Sigma_alu = 255*1e6     #Pa  --> try to link a material database to the code
    Density_alu = 2700      #kg/m3 --> try to link a material database to the code
    Sigma_shear_alu = 150*1e6   #Pa
    r_shear = 0.130/2           #m
    V_ullage = 0.1              #amount of the total liquid volume, used for ullage
    P_SF_Metallic = SF_metallic*Pmax
    P_SF_composite = SF_composite*Pmax   #look into incorporating this in the profile of the engine

    #Thickness wall
    if Profile["tank_material"] == "metal":
        t_wall_1 = (P_SF_Metallic*(Dtank/2))/Sigma_alu
        t_wall_2 = (P_SF_Metallic*(Dtank/2))/(Sigma_alu*2)
    elif Profile["tank_material"] =="composite":
        t_wall_1 = (P_SF_composite*(Dtank/2))/Sigma_alu
        t_wall_2 = (P_SF_composite*(Dtank/2))/(Sigma_alu*2)

    #Thickness bulkheads

    t_bulkhead = ((np.pi/4)*Dtank**2*P_SF_Metallic)/(Sigma_shear_alu*2*np.pi*r_shear)

    #Length cylinder tank

    

    #Mass tank
    #Volume cylinder
    D_outer_cylinder = Dtank+2*t_wall_1
    
    
    for i in range(len(OF_prof["C*"])):
        m_dot = Avg_Thrust/(OF_prof["C*"][i]*OF_prof["Cf"][i])
        m_dot_list.append(m_dot)
        def t_burn(tb):
            v_bo = Perf_eff*(Avg_Thrust/(m_rocket-(m_dot*tb)) -g0*np.sin(Launch_angle*np.pi/180))*tb
            t_ap = v_bo*np.sin(Launch_angle*np.pi/180)/g0 + tb
            h_bo = (1/2)/(Avg_Thrust*np.sin(Launch_angle*np.pi/180)/(m_rocket-(m_dot*tb)) - g0)*tb**2
            return Perf_eff*(h_bo + v_bo *np.sin(Launch_angle*np.pi/180)*(t_ap-tb) -(1/2)*g0*(t_ap-tb)**2 ) - TgtAlt
        t_b = float(fsolve(t_burn,0.2)[0])
        tb_list.append(t_b)
        v_bo = Perf_eff*(Avg_Thrust/(m_rocket-(m_dot*t_b)) -g0*np.sin(Launch_angle*np.pi/180))*t_b
        t_ap = v_bo*np.sin(Launch_angle*np.pi/180)/g0 + t_b
        h_bo = (1/2)/(Avg_Thrust*np.sin(Launch_angle*np.pi/180)/(m_rocket-(m_dot*t_b)) - g0)*t_b**2
        h = Perf_eff*(h_bo + v_bo *np.sin(Launch_angle*np.pi/180)*(t_ap-t_b) -(1/2)*g0*(t_ap-t_b)**2 )
        Alt_list.append(h)
        m_prop_list.append(t_b*m_dot)
        m_ox_list.append((t_b*m_dot)*OF_prof["OF"][i]/(OF_prof["OF"][i]+1))
        m_fuel_list.append((t_b*m_dot)*1/(OF_prof["OF"][i]+1))
        I_tot_list.append(Avg_Thrust*t_b)
        V_ox.append(m_ox_list[i]/Rho_Ox_L)
        L_tank.append((4*(1+V_ullage)*V_ox[i])/(np.pi*Dtank**2))
        V_cylinder = L_tank[i]*(((np.pi*D_outer_cylinder**2)/4)-((np.pi*Dtank**2)/4))
        M_cylinder = V_cylinder*Density_alu
        #Volume bulkheads
        V_bulkhead = t_bulkhead*(np.pi*Dtank**2)/4
        M_bulkhead = V_bulkhead*Density_alu
        #Total mass tank
        M_tank.append(M_cylinder+2*M_bulkhead)
        M_system.append(M_tank[i] + m_prop_list[i])
        
    
    
    fig, ax1 = plt.subplots()
    plt.grid()
    ax2 = ax1.twinx()
    ax1.plot(OF_prof["OF"],m_dot_list,color='blue',label="m_dot [kg/s]")
    ax2.plot(OF_prof["OF"],tb_list,color='red',label="burn time [s]")
    ax1.set_xlabel("O/F ratio")
    ax1.set_xticks(np.arange(3,13,1))
    ax1.set_ylabel("massflow [kg/s]")
    ax2.set_ylabel("burn time [s]", color ='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.spines['right'].set_color('red')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.suptitle(f"massflow & burn time vs O/F ({Profile['Fuel']} & {Profile['Oxidizer']})", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    fig, ax1 = plt.subplots()
    plt.grid()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.2))
    ax1.plot(OF_prof["OF"],m_ox_list,color='blue',label="m_ox [kg]")
    ax1.plot(OF_prof["OF"],m_prop_list,color='green',label="m_propellant [kg]")
    ax2.plot(OF_prof["OF"],m_fuel_list,color='red',label="m_fuel [kg]")
    ax3.plot(OF_prof["OF"],M_system ,color='purple',label="M_system [kg]")
    ax1.set_xlabel("O/F ratio")
    ax1.set_xticks(np.arange(3,13,1))
    ax1.set_ylabel("mass [kg]")
    ax2.set_ylabel("fuel mass [kg]", color ='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.spines['right'].set_color('red')
    ax3.set_ylabel("System mass [kg]",color ='purple')
    ax3.tick_params(axis='y', colors='purple')
    ax3.spines['right'].set_color('purple')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.suptitle(f"m_prop vs O/F ({Profile['Fuel']} & {Profile['Oxidizer']})", fontsize=12)
    plt.tight_layout()
    plt.show()
        
    """
    plt.title("Altitude check")  #added for performing checks
    plt.plot(OF_prof["OF"],Alt_list)
    plt.show()
    """

    
    
    fig, ax1 = plt.subplots()
    plt.grid()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.2))
    ax1.plot(OF_prof["OF"],m_ox_list,color='blue',label="m_ox [kg]")
    ax1.plot(OF_prof["OF"],m_prop_list,color='green',label="m_propellant [kg]")
    ax2.plot(OF_prof["OF"],V_ox,color='red',label="V tank [m³]")
    ax3.plot(OF_prof["OF"],M_tank ,color='green',label="M_tank [kg]")
    ax1.set_xlabel("O/F ratio")
    ax1.set_xticks(np.arange(3,13,1))
    ax1.set_ylabel("mass [kg]")
    ax2.set_ylabel("Tank volume [m³]", color ='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.spines['right'].set_color('red')
    ax3.set_ylabel("Tank mass [kg]",color ='green')
    ax3.tick_params(axis='y', colors='green')
    ax3.spines['right'].set_color('green')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.suptitle(f"m_prop & tank vol vs O/F ({Profile['Fuel']} & {Profile['Oxidizer']})", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    

    OF_opt = OF_prof["OF"][np.argmin(m_prop_list)]
    OF_opt_mass = OF_prof["OF"][np.argmin(M_system)]
    m_fuel_opt = m_fuel_list[np.argmin(M_system)]
    L_tank_opt = L_tank[np.argmin(M_system)]
    M_tank_opt = M_tank[np.argmin(M_system)]
    Profile["tb"] = tb_list[np.argmin(M_system)]
    Profile["OF"] = OF_opt
    Profile["DesignThrust"] = Avg_Thrust
    Profile["m_fuel"] = m_fuel_opt
    Profile["m_tank"] = M_tank_opt
    
    print("Average Thrust [N]:",Avg_Thrust)
    print("Optimum O/F:" , OF_opt)
    print("Optimum O/F system mass:", OF_opt_mass)
    print("Burn time [s]:", Profile["tb"])
    print("tank mass [kg]:",Profile["m_tank"])
    print("L tank opt [m]:",L_tank_opt)
    return{"OF_opt":OF_opt,"m_fuel":m_fuel_opt,"L_opt":L_tank_opt,"L_tank":(L_tank_opt + 2*t_bulkhead),"Dtank":Dtank,"t_tank":t_wall_1}
    


#%% CHAMBER GEOMETRY

def CH_Geo(Profile, Nozzle_Type, Ac_At, f_pct,Phi_i, Phi_n, Phi_e, export,plotEnable, Grid = "Coarse"):
    
    gamma_c = CEA_RUN(Profile)["gamma_c"]
    print("gamma_c",gamma_c)
    gamma_t = CEA_RUN(Profile)["gamma_t"]
    Rho_c = CEA_RUN(Profile)["Rho_c"]
    Tc = CEA_RUN(Profile)["Tc"]
    R_spec = CEA_RUN(Profile)["Rspec"]
    a_t = CEA_RUN(Profile)["a_c"]
    Pc = CEA_RUN(Profile)["Pc"]
    #flow divergence losses calculation
    if Nozzle_Type == "Conical":
        e_div = 1 - (1+np.cos(Phi_e*np.pi/180))/2
    elif Nozzle_Type =="Bell":
        e_div = 0.5*(1-np.cos(((Phi_n+Phi_e)*np.pi/180)/2))
    V_c = 1/Rho_c  #Chamber specific volume [m³/kg]
    V_t = V_c * ((gamma_c + 1)/2)**(1/(gamma_c - 1))
    T_t = 2*Tc/(gamma_t + 1)
    v_t = np.sqrt(gamma_t * R_spec * Tc)
    v_e = np.sqrt((2*gamma_c)/(gamma_c - 1)* R_spec * Tc*(1-((Profile["Popt"]/1e5)/Profile["DesignChamberPressure"])**((gamma_c - 1)/gamma_c)))
    m_tot = Profile["DesignThrust"]*(1+e_div)/v_e     #Mass flow rate for the required thrust
    print("m_tot",m_tot,"kg/s")
    At = m_tot * V_t/a_t 
    Dt = np.sqrt(At*4/(np.pi))
    print("Dt", Dt , "m")
    Ae = At * (((gamma_c + 1)/2)**(1/(gamma_c - 1)) * ((Profile["Popt"]/1e5)/Profile["DesignChamberPressure"])**(1/gamma_c)*np.sqrt((gamma_c + 1)/(gamma_c - 1) * (1-((Profile["Popt"]/100000)/Profile["DesignChamberPressure"])**((gamma_c - 1)/gamma_c) )))**(-1)
    De = np.sqrt(Ae*4/(np.pi))
    print("De", De , "m")
    Ac = At * Ac_At
    Dc = np.sqrt(Ac*4/(np.pi))
    Ae_At = Ae/At
    print("Ae_At", Ae_At)
    print("Dc", Dc , "m")
    #%% Cylindrical chamber geometry
        #%% chamber wall thickness
    SF_metallic = 4         #EuRoC is 2
    SF_composite = 3        #EuRoC is 3
    Sigma_alu = 255*1e6     #Pa  --> try to link a material database to the code
    Density_alu = 2700      #kg/m3 --> try to link a material database to the code
    Sigma_shear_alu = 150*1e6   #Pa  
    r_shear = Dc/2           #m
        
    P_SF_Metallic = SF_metallic*Pc*1e5
    P_SF_composite = SF_composite*Pc*1e5   #look into incorporating this in the profile of the engine
    if Profile["tank_material"] == "metal":
        t_wall = (P_SF_Metallic*(Dc/2))/Sigma_alu
    elif Profile["tank_material"] =="composite":
        t_wall= (P_SF_composite*(Dc/2))/Sigma_alu

        #%%Thickness injector plate

    t_bulkhead = ((np.pi/4)*Dc**2*P_SF_Metallic)/(Sigma_shear_alu*2*np.pi*r_shear)
    
    #%% VAPORIZATION CALCULATION
    
    m_ox = Profile["OF"] * m_tot /(Profile["OF"]+1)     #determining oxidizer mass flow
    Profile["m_o_dot"] = m_ox
    m_f = m_tot /(Profile["OF"]+1)     #determining fuel mass flow
    Ai_o = m_ox/ Profile["Cdi"] * 1/(np.sqrt(2*CEA_RUN(Profile)["RhooG"]*Profile["DPi"]*Pc*1e5))
    Ui_o = m_ox/(Ai_o * CEA_RUN(Profile)["RhooG"])  #discharge velocity from injector
    
    if Profile["InjectorType"] == "Shower-Head":
        Di_o = np.sqrt(Ai_o*4 / (np.pi * Profile["n_inj"])) #orifice diameter, based on nr of elements
        print("Di_o", Di_o)

    #%% Grain calculations
    steps = 100
    t_list = np.arange(0, Profile["tb"], Profile["tb"]/steps)  # creates a time array
    Df = Dc
    Di = Df / 3  # rule of thumb for the diameter ratio of grain
    # Initial guess
    D_i_guess = 0.001
    tolerance = 0.01
    max_iterations = 1000

    for i in range(max_iterations):
        L_grain = (Profile["m_fuel"] / Profile["Rho_f"]) / ((Df**2 - D_i_guess**2) * np.pi / 4)
        OF_init = Profile["m_o_dot"] / ((Profile["Rho_f"] * (Profile["a0"] * (Profile["m_o_dot"] / (D_i_guess**2 * np.pi / 4))**Profile["k"]) * 1e-3 * np.pi * D_i_guess * L_grain))
        Residual = np.abs(Profile["OF"] - OF_init)

        #print("Residual:", Residual)
        #print("iter:", i)

        if Residual < tolerance:
            break
        if Residual > 0.5:
            D_i_guess = D_i_guess + 1e-3
        else:
            D_i_guess = D_i_guess + 1e-4

    # Print the final results
    print("The port diameter [m]:", D_i_guess)
    print("The Grain length is [m]:", L_grain)
    print("Profile OF", Profile["OF"])
    print("OF_initial", OF_init)
    Dp = D_i_guess
    Lg = L_grain
    #%% Pre-and Post-CC
    L_post = 0.5*Dc  #For now just a simple rule of thumb 0.5 (0.1)
    L_pre = (3/4)*Dc
    L_tot = L_post + L_pre + Lg
    #%% Chamber mass
    V_chamber = L_tot * ((Dc + 2*t_wall)**2 - Dc**2) *np.pi/4
    Dch = Dc + 2*t_wall
    M_ch_wall = V_chamber *  Density_alu
    M_ch = M_ch_wall + ((Dc + 2*t_wall)**2 *np.pi/4) *t_bulkhead*Density_alu
    print("Chamber length [m]:",L_tot)
    print("Chamber wall thickness [m]:",t_wall)
    print("Bulkhead thickness [m]:",t_bulkhead)
    print("Chamber mass [kg]:",M_ch)
    #%% NOZZLE TYPE

    if Nozzle_Type == 'Conical' : 
        Ln_cone = (np.sqrt(Ae_At)-1)*Dt/(2*np.tan(Phi_e*np.pi/180)) #determines the length of the cone from the throat
        x_exit = np.arange(0,Ln_cone,Ln_cone/20 )
        y_exit = Dt/2 + x_exit*np.tan(Phi_e*np.pi/180)
        Phirange1 = np.arange(-90-Phi_i , -90 , 0.1)
        x_entr = 1.5*Dt/2*np.cos(Phirange1 * np.pi/180)
        y_entr = 1.5*Dt/2*np.sin(Phirange1 * np.pi/180)+ 1.5*Dt/2 + Dt/2
        Phirange3 = np.arange(0,Phi_i,0.5)
        y_c0 = (1.5*(Dt/2)*np.sin((-90-Phi_i) * np.pi/180)+ 1.5*Dt/2 + Dt/2)
        R_c = (Dc/2 - y_c0 )/(1-np.cos(Phi_i * np.pi/180))
        Offset = R_c - Dc/2
        x_c = np.sin((Phirange3) * np.pi/180)*R_c -R_c*np.sin(Phi_i * np.pi/180) + 1.5*Dt/2*np.cos((90+Phi_i) * np.pi/180)
        y_c =  R_c * np.cos((Phirange3) * np.pi/180) - Offset 
        y_c0 = (1.5*(Dt/2)*np.sin((-90-Phi_i) * np.pi/180)+ 1.5*Dt/2 + Dt/2)
        R_c = (Dc/2 - y_c0 )/(1-np.cos(Phi_i * np.pi/180))
        Offset = R_c - Dc/2
        x_inlet = np.arange(np.sin((0) * np.pi/180)*R_c -R_c*np.sin(Phi_i * np.pi/180) + 1.5*Dt/2*np.cos((90+Phi_i) * np.pi/180) - (4/3)*L_post,1.5*Dt/2*np.cos(-90-Phi_i * np.pi/180) + np.cos((90 + Phi_i) * np.pi/180)*R_c, (4/3)*L_post/2) 
        y_inlet = [Dc/2] * len(x_inlet)
        x_coord = []
        y_coord = []
            
        x_combined = np.concatenate((np.append(x_coord, x_c),np.append(x_coord, x_entr),
                                         np.append(x_coord, x_exit) ,np.append(x_coord,x_inlet)))
            #,np.append(x_coord,x_inlet)

        y_combined = np.concatenate((np.append(y_coord, y_c), np.append(y_coord, y_entr),
                                         np.append(y_coord, y_exit),np.append(y_coord,y_inlet)))
            #,np.append(y_coord,y_inlet)
        ContourCoordInt = pd.DataFrame({"x": x_combined ,"y": y_combined,"z": [0]*len(x_combined)})
        ContourCoordInt = ContourCoordInt.sort_values("x",ignore_index=True)
        ContourCoordInt = ContourCoordInt.drop_duplicates()
        
    elif Nozzle_Type == 'Bell' :
        Ln = f_pct * ((np.sqrt(Ae_At)-1)*Dt/(2*np.tan(15*np.pi/180)))
        print("Ln", Ln , "m")
        #entrance section
        Phirange1 = np.arange(-90-Phi_i , -90 , 1)
        x_entr = 1.5*Dt/2*np.cos(Phirange1 * np.pi/180)
        y_entr = 1.5*Dt/2*np.sin(Phirange1 * np.pi/180)+ 1.5*Dt/2 + Dt/2
        #exit section
        Phirange2 = np.arange(-90 , Phi_e-90 , 1)
        x_exit = 0.5*Dt/2*np.cos(Phirange2 * np.pi/180)       #originally aft radius was 0.382
        y_exit = 0.5*Dt/2*np.sin(Phirange2 * np.pi/180)+0.5*Dt/2 + Dt/2
        Nx = 0.5*Dt/2*np.cos((Phi_e-90) * np.pi/180)
        Ny = 0.5*Dt/2*np.sin((Phi_e-90)* np.pi/180)+0.5*Dt/2 + Dt/2
        #Quadratic Bézier parabola curve
        m1 = np.tan(Phi_n*np.pi/180)
        m2 = np.tan(Phi_e*np.pi/180)
        C1 = Ny - m1 * Nx
        C2 = De/2 - m2 * Ln
        Qx = (C2-C1)/(m1-m2)
        Qy = m1*Qx+C1
        t_range = np.arange(0,1,0.05)
        x_t = (1-t_range)**2 * Nx + 2*(1-t_range)*t_range * Qx + t_range**2 * Ln
        y_t = (1-t_range)**2 * Ny + 2*(1-t_range)*t_range * Qy + t_range**2 * De/2
        Phirange3 = np.arange(0,Phi_i,0.5)
        y_c0 = (1.5*(Dt/2)*np.sin((-90-Phi_i) * np.pi/180)+ 1.5*Dt/2 + Dt/2)
        R_c = (Dc/2 - y_c0 )/(1-np.cos(Phi_i * np.pi/180))
        Offset = R_c - Dc/2
        x_c = np.sin((Phirange3) * np.pi/180)*R_c -R_c*np.sin(Phi_i * np.pi/180) + 1.5*Dt/2*np.cos((90+Phi_i) * np.pi/180)
        y_c =  R_c * np.cos((Phirange3) * np.pi/180) - Offset 
        x_inlet = np.arange(np.sin((0) * np.pi/180)*R_c -R_c*np.sin(Phi_i * np.pi/180) + 1.5*Dt/2*np.cos((90+Phi_i) * np.pi/180) - (4/3)*L_post,1.5*Dt/2*np.cos(-90-Phi_i * np.pi/180) + np.cos((90 + Phi_i) * np.pi/180)*R_c, (4/3)*L_post/2) 
        y_inlet = [Dc/2] * len(x_inlet)
        
        x_coord = []
        y_coord = []
        x_condi = []
        y_condi = []
            
        x_combined = np.concatenate((np.append(x_coord, x_c),np.append(x_coord, x_entr),
                                         np.append(x_coord, x_exit), np.append(x_coord, x_t) ,np.append(x_coord,x_inlet)))
            #,np.append(x_coord,x_inlet)

        y_combined = np.concatenate((np.append(y_coord, y_c), np.append(y_coord, y_entr),
                                         np.append(y_coord, y_exit), np.append(y_coord, y_t),np.append(y_coord,y_inlet)))
        
        y_combinedsup = [x + Profile["wall thickness"]*10**(-3) for x in y_combined]

            #,np.append(y_coord,y_inlet)
        ContourCoordInt = pd.DataFrame({"x": x_combined ,"y": y_combined,"z": [0]*len(x_combined)})
        ContourCoordIntsup=pd.DataFrame({"x": x_combined ,"y": y_combinedsup,"z": [0]*len(x_combined)})
        ContourCoordInt = ContourCoordInt.sort_values("x",ignore_index=True)
        ContourCoordIntsup = ContourCoordIntsup.sort_values("x",ignore_index=True)
        ContourCoordInt = ContourCoordInt.drop_duplicates()
        ContourCoordIntsup = ContourCoordIntsup.drop_duplicates()


        if export == "OF":
            x_combined1 = np.concatenate((np.append(x_condi, x_c),np.append(x_condi, x_entr),
                                         np.append(x_condi, x_exit), np.append(x_condi, x_t)))
            #,np.append(x_coord,x_inlet)

            y_combined1 = np.concatenate((np.append(y_coord, y_c), np.append(y_coord, y_entr),
                                            np.append(y_coord, y_exit), np.append(y_coord, y_t)))
                #,np.append(y_coord,y_inlet)
            ContourCoordOF = pd.DataFrame({"x": x_combined1 ,"y": y_combined1,"z": [0]*len(x_combined1)})
            ContourCoordOF = ContourCoordOF.sort_values("x",ignore_index=True)
            ContourCoordOF = ContourCoordOF.drop_duplicates()
            if Grid == "Coarse":
                ScaleF = 5/2
            elif Grid == "Medium":
                ScaleF = 5/3
            elif Grid == "Fine":
                ScaleF == 5/4
            elif Grid == "Ultra Fine":
                ScaleF == 1

            with open("Nozzle", 'w') as file:
                #file.write("convertToMeters 0.001; //all dimensions are in mm this way \n")
                file.write("vertices\n(\n")
                file.write(f"({x_inlet[0]}  0  0.0005) //0\n")
                file.write(f"({x_inlet[0]} {y_inlet[0]} 0.0005) //1\n")
                file.write(f"({x_c[0]} {Dc/2} 0.0005) //2\n")
                file.write(f"({x_t[-1]} {y_t[-1]} 0.0005) //3\n")
                file.write(f"({x_t[-1]} 0  0.0005) //4\n")
                file.write(f"({x_c[0]} 0  0.0005) //5\n \n")

                file.write(f"({x_inlet[0]}  0  -0.0005) //6\n")
                file.write(f"({x_inlet[0]} {y_inlet[0]} -0.0005) //7\n")
                file.write(f"({x_c[0]} {Dc/2} -0.0005) //8\n")
                file.write(f"({x_t[-1]} {y_t[-1]} -0.0005) //9\n")
                file.write(f"({x_t[-1]} 0  -0.0005) //10\n")
                file.write(f"({x_c[0]} 0  -0.0005) //11\n \n")

                file.write(f"(0.500  {y_t[-1]} 0.0005) //12 \n")
                file.write(f"(0.500  0 0.0005) //13 \n")
                file.write(f"({x_t[-1]}  0.100 0.0005) //14 \n")
                file.write(f"(0.500  0.100 0.0005) //15 \n")
                file.write(f"(0.500  {y_t[-1]} -0.0005) //16 \n")
                file.write(f"(0.500  0 -0.0005) //17 \n")
                file.write(f"({x_t[-1]}  0.100 -0.0005) //18 \n")
                file.write(f"(0.500  0.100 -0.0005) //19 \n")
                file.write(f"(0.120 {y_t[-1]} 0.0005) //20 \n")
                file.write(f"(0.120 0.100 0.0005) //21 \n")
                file.write(f"(0.120 {y_t[-1]} -0.0005) //22 \n")
                file.write(f"(0.120 0.100 -0.0005) //23 \n")


                file.write(");\n\nblocks\n(\n")
                # Example: Write a single block definition using the vertices
                # Adjust this according to your actual block structure
                file.write(f"    hex (0 1 2 5 6 7 8 11) ({int(150/ScaleF)} {int(25/ScaleF)} 1) simpleGrading (0.1 0.1 1) //block 0 \n")
                file.write(f"    hex (5 2 3 4 11 8 9 10) ({int(150/ScaleF)} {int(250/ScaleF)} 1) simpleGrading (0.1 0.2 1) //block 1 \n")
                file.write(f"    hex (4 3 12 13 10 9 16 17) ({int(150/ScaleF)} {int(250/ScaleF)} 1) simpleGrading (0.1 25 1) //block 2 \n")
                file.write(f"    hex (3 14 15 12 9 18 19 16) ({int(120/ScaleF)} {int(250/ScaleF)} 1) simpleGrading (50 25 1) //block 3 \n")
                #file.write(f"    hex (9 22 23 18 3 20 21 14) ({20/ScaleF} {120/ScaleF} 1) simpleGrading (30 50 1) //block 4 \n")
                file.write(");\n\nedges\n(\n")
                # Example: Write edges using vertex indices for curves
                # Adjust this according to your actual curves
                file.write("polyLine 2 3 \n ( \n")
                for i in range(len(ContourCoordOF) - 1):
                    file.write(f"({ContourCoordOF.iloc[i]['x']} {ContourCoordOF.iloc[i]['y']} 0.0005)\n")
                file.write(") \n")
                file.write("polyLine 8 9 \n ( \n")
                for i in range(len(ContourCoordOF) - 1):
                    file.write(f"({ContourCoordOF.iloc[i]['x']} {ContourCoordOF.iloc[i]['y']} -0.0005)\n")
                file.write(") \n")
                
                file.write(");\n\nboundary\n(\n")
                # Define boundary patches (e.g., inlet, outlet, walls)
                file.write("    inlet\n    {\n        type patch;\n        faces\n        (\n            (0 1 7 6)\n        );\n    }\n")
                file.write("    outlet-1\n    {\n        type patch;\n        faces\n        (\n            (12 13 17 16)\n (15 12 16 19)\n        );\n    }\n")
                file.write("    outlet-2\n    {\n        type patch;\n        faces\n        (\n             (14 15 19 18)\n        );\n    }\n")
                file.write("    nozzle\n    {\n        type wall;\n        faces\n        (\n            (1 2 8 7)\n (2 3 9 8)\n (3 14 18 9)\n );\n    }\n")
                file.write("    bottom\n    {\n        type symmetryPlane;\n        faces\n        (\n            (0 6 11 5)\n (5 11 10 4)\n (4 10 17 13)\n       );\n    }\n")
                file.write(");\n \n // ************************************************************************* //")

        if export == "OF_axs":
            x_combined1 = np.concatenate((np.append(x_condi, x_c),np.append(x_condi, x_entr),
                                         np.append(x_condi, x_exit), np.append(x_condi, x_t)))
            #,np.append(x_coord,x_inlet)

            y_combined1 = np.concatenate((np.append(y_coord, y_c), np.append(y_coord, y_entr),
                                            np.append(y_coord, y_exit), np.append(y_coord, y_t)))
                #,np.append(y_coord,y_inlet)
            ContourCoordOF = pd.DataFrame({"x": x_combined1 ,"y": y_combined1,"z": [0]*len(x_combined1)})
            ContourCoordOF = ContourCoordOF.sort_values("x",ignore_index=True)
            ContourCoordOF = ContourCoordOF.drop_duplicates()
            if Grid == "Coarse":
                ScaleF = 5/2
            elif Grid == "Medium":
                ScaleF = 5/3
            elif Grid == "Fine":
                ScaleF == 5/4
            elif Grid == "Ultra Fine":
                ScaleF == 1

            wedge_angle = 2.5 * np.pi/180
            L_ch = 0.5
            H_ch = 5 * De

            with open("Nozzle_axs", 'w') as file:
                #file.write("convertToMeters 0.001; //all dimensions are in mm this way \n")
                file.write("vertices\n(\n")
                file.write(f"({x_inlet[0]}  0  0) //0\n")
                file.write(f"({x_inlet[0]} {y_inlet[0]} {y_inlet[0]*np.tan(wedge_angle/2)}) //1\n")
                file.write(f"({x_c[0]} {Dc/2} {y_c[0]*np.tan(wedge_angle/2)}) //2\n")
                file.write(f"({x_t[-1]} {y_t[-1]} {y_t[-1]*np.tan(wedge_angle/2)}) //3\n")
                file.write(f"({x_t[-1]} 0  0) //4\n")
                file.write(f"({x_c[0]} 0  0) //5\n \n")

                
                file.write(f"({x_inlet[0]} {y_inlet[0]} {-1 * y_inlet[0]*np.tan(wedge_angle/2)}) //6\n")
                file.write(f"({x_c[0]} {Dc/2} {-1 * y_c[0]*np.tan(wedge_angle/2)}) //7\n")
                file.write(f"({x_t[-1]} {y_t[-1]} {-1 * y_t[-1]*np.tan(wedge_angle/2)}) //8\n")
                
                file.write(f"({L_ch}  {y_t[-1]} {y_t[-1]*np.tan(wedge_angle/2)}) //9 \n")
                file.write(f"({L_ch}  0 0) //10 \n")
                file.write(f"({x_t[-1]}  {H_ch} {H_ch*np.tan(wedge_angle/2)}) //11 \n")
                file.write(f"({L_ch}  {H_ch} {H_ch*np.tan(wedge_angle/2)}) //12 \n")
                file.write(f"({L_ch}  {y_t[-1]} {-1 * y_t[-1]*np.tan(wedge_angle/2)}) //13 \n")
                
                file.write(f"({x_t[-1]}  {H_ch} {-1 * H_ch*np.tan(wedge_angle/2)} ) //14 \n")
                file.write(f"({L_ch}  {H_ch}  {-1 * H_ch*np.tan(wedge_angle/2)}) //15 \n")
                


                file.write(");\n\nblocks\n(\n")
                # Example: Write a single block definition using the vertices
                # Adjust this according to your actual block structure
                file.write(f"    hex (0 6 1 0 5 7 2 5) ({int(150/ScaleF)} 1 {int(25/ScaleF)}) simpleGrading (0.1 1 0.1) //block 0 \n")
                file.write(f"    hex (5 7 2 5 4 8 3 4) ({int(150/ScaleF)} 1 {int(250/ScaleF)}) simpleGrading (0.1 1 0.2) //block 1 \n")
                file.write(f"    hex (4 8 3 4 10 13 9 10) ({int(150/ScaleF)} 1 {int(250/ScaleF)}) simpleGrading (0.1 1 25) //block 2 \n")
                file.write(f"    hex (3 11 12 9 8 14 15 13) ({int(120/ScaleF)} {int(250/ScaleF)} 1) simpleGrading (50 25 1) //block 3 \n")
                #file.write(f"    hex (9 22 23 18 3 20 21 14) ({20/ScaleF} {120/ScaleF} 1) simpleGrading (30 50 1) //block 4 \n")
                file.write(");\n\nedges\n(\n")
                # Example: Write edges using vertex indices for curves
                # Adjust this according to your actual curves
                file.write("polyLine 2 3 \n ( \n")
                for i in range(len(ContourCoordOF) - 1):
                    file.write(f"({ContourCoordOF.iloc[i]['x']} {ContourCoordOF.iloc[i]['y']} {ContourCoordOF.iloc[i]['y']*np.tan(wedge_angle/2)})\n")
                file.write(") \n")
                file.write("polyLine 7 8 \n ( \n")
                for i in range(len(ContourCoordOF) - 1):
                    file.write(f"({ContourCoordOF.iloc[i]['x']} {ContourCoordOF.iloc[i]['y']} {-1 * ContourCoordOF.iloc[i]['y']*np.tan(wedge_angle/2)})\n")
                file.write(") \n")
                
                file.write(");\n\nboundary\n(\n")
                # Define boundary patches (e.g., inlet, outlet, walls)
                file.write("    inlet\n    {\n        type patch;\n        faces\n        (\n            (0 1 6 0)\n        );\n    }\n")
                file.write("    outlet-1\n    {\n        type patch;\n        faces\n        (\n            (10 13 9 10)\n (12 9 13 15)\n        );\n    }\n")
                file.write("    outlet-2\n    {\n        type patch;\n        faces\n        (\n             (11 12 15 14)\n        );\n    }\n")
                file.write("    nozzle\n    {\n        type wall;\n        faces\n        (\n            (1 2 7 6)\n (2 3 8 7)\n (3 11 14 8)\n );\n    }\n")
                file.write("    wedge1\n    {\n        type wedge;\n        faces\n        (\n            (0 5 2 1)\n (5 4 3 2)\n (4 10 9 3)\n (3 9 12 11)\n );\n    }\n")
                file.write("    wedge2\n    {\n        type wedge;\n        faces\n        (\n            (0 6 7 5)\n (5 7 8 4)\n (4 8 13 10)\n (8 14 15 13)\n   );\n    }\n")
                file.write(");\n \n // ************************************************************************* //")

        
    #%% SAVING AND PLOTTING COORDINATES

    if plotEnable == 'True':
        plt.title('Nozzle contour  ' + Profile["name"])
        plt.grid()
        #plt.xlim(, Ln)
        if Dc > De:
            plt.ylim(0, Dc/2 + Dc/2*0.2)
        else:
            plt.ylim(0, De/2 + De/2*0.1)
        plt.plot(ContourCoordInt["x"],ContourCoordInt["y"],label="inner contour")
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel("Axial distance, 0 = throat [m]")
        plt.ylabel("Radius [m]")
        plt.show()
        if Profile["wall thickness"]!=0:
            plt.plot(ContourCoordInt["x"], ContourCoordInt["y"], label="inner contour", color='black')
            plt.plot(ContourCoordIntsup["x"], ContourCoordIntsup["y"], label="outer contour", color='black')

    # Ensure that the contour data exists
        if not ContourCoordInt.empty and not ContourCoordIntsup.empty:
        # Get first and last points for both contours safely
            x_start_inner, y_start_inner = ContourCoordInt["x"].iloc[0], ContourCoordInt["y"].iloc[0]
            x_end_inner, y_end_inner = ContourCoordInt["x"].iloc[-1], ContourCoordInt["y"].iloc[-1]

            x_start_outer, y_start_outer = ContourCoordIntsup["x"].iloc[0], ContourCoordIntsup["y"].iloc[0]
            x_end_outer, y_end_outer = ContourCoordIntsup["x"].iloc[-1], ContourCoordIntsup["y"].iloc[-1]

        # Connect start points with a vertical line
            plt.plot([x_start_inner, x_start_outer], [y_start_inner, y_start_outer], color='black')

        # Connect end points with a vertical line
            plt.plot([x_end_inner, x_end_outer], [y_end_inner, y_end_outer], color='black')


            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
        
            plt.xlabel("Axial distance, 0 = throat [m]")
            plt.ylabel("Radius [m]")
            plt.legend()
            plt.show()
         
    if export == "CSV":

        ContourCoord_exp = ContourCoordInt * 1e2   #1e2 because Fusion CSV reader works with cm
        ContourCoord_exp.to_csv(f'contour{Profile["name"]}.csv', index=False)
            
        #elif export == "OpenFOAM":
    elif export == "GMSH":
        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("nozzle")

        # Create points based on your coordinates
        ContourCoordInt = pd.DataFrame({"x": x_combined, "y": y_combined, "z": [0] * len(x_combined)})
        ContourCoordInt = ContourCoordInt.sort_values("x", ignore_index=True)
        ContourCoordInt = ContourCoordInt.drop_duplicates()

        for i, row in ContourCoordInt.iterrows():
            gmsh.model.geo.addPoint(row['x'], row['y'], row['z'], 1.0, i+1)

        # Create lines between points
        for i in range(len(ContourCoordInt) - 1):
            gmsh.model.geo.addLine(i + 1, i + 2, i + 1)
        gmsh.model.geo.addLine(len(ContourCoordInt), 1, len(ContourCoordInt))

        # Create curve loop and plane surface
        curve_tags = [i + 1 for i in range(len(ContourCoordInt))]
        gmsh.model.geo.addCurveLoop(curve_tags, 1)
        gmsh.model.geo.addPlaneSurface([1], 1)

        # Synchronize the model
        gmsh.model.geo.synchronize()

        # Generate the mesh
        gmsh.model.mesh.generate(2)

        # Save the mesh to a file
        gmsh.write("nozzle.msh")

        # Finalize gmsh
        gmsh.finalize()

    elif export == "FreeCad":
    # Create a new DXF document
        doc = ezdxf.new()
        msp = doc.modelspace()

        # Prepare points for the inner contour spline
        inner_points = [(row["x"], row["y"]) for _, row in ContourCoordInt.iterrows()]
        outer_points = [(row["x"], row["y"]) for _, row in ContourCoordIntsup.iterrows()]

        # Add spline for the inner contour
        if len(inner_points) > 1:  # Ensure there are enough points for a spline
            msp.add_spline(inner_points, degree=3)  # Cubic spline (degree=3) for smoothness

        # Add spline for the outer contour
        if len(outer_points) > 1:
            msp.add_spline(outer_points, degree=3)

        # Connect the start and end points of inner and outer contours with lines
        x_start_inner, y_start_inner = inner_points[0]
        x_end_inner, y_end_inner = inner_points[-1]
        x_start_outer, y_start_outer = outer_points[0]
        x_end_outer, y_end_outer = outer_points[-1]

        msp.add_line((x_start_inner, y_start_inner), (x_start_outer, y_start_outer))  # Start connection
        msp.add_line((x_end_inner, y_end_inner), (x_end_outer, y_end_outer))          # End connection

        # Save DXF file
        doc.saveas("nozzle_contour.dxf")




    #%% Temperature, pressure and velocity across nozzle
    Vandk = np.sqrt(gamma_c) * (2/(gamma_c + 1))**((gamma_c+1)/(2*(gamma_c-1)))
    Mx_list=[]
    Px_list =[]
    Tx_list = []
    Rho_x_list = []
    #print("contourcoordInternal",ContourCoordInt)
    for i in range(len(ContourCoordInt["y"])):
        def M_solve(M_x):
            return (1/M_x) * np.sqrt(((1+(gamma_c-1)/2 * M_x**2)/(1+(gamma_c - 1)/2))**((gamma_c + 1)/(gamma_c-1)))- (np.pi * ContourCoordInt["y"][i]**2)/At
        if ContourCoordInt["x"][i] < 0:
            Mx =(fsolve(M_solve, 0.05 ))
            Mx_min = min(Mx)
            Mx_list.append(Mx_min)
        elif ContourCoordInt["x"][i] >= 0:
            Mx =(fsolve(M_solve, 2.5 ))
            Mx_max = max(Mx)
            Mx_list.append(Mx_max)
             
        Px_list.append(Pc*1e5/(1+(gamma_c-1)/2 * Mx_list[i]**2)**(gamma_c/(gamma_c-1)))
        Tx_list.append(Tc*(Px_list[i]/(Pc*1e5))**((gamma_c-1)/gamma_c))
        Rho_x_list.append(Rho_c*(Px_list[i]/(Pc*1e5))**(1/gamma_c))
    
    #print(Mx_list)   
    #print(Tx_list)
    #print(type(Tx_list))
    #print(type(Mx_list))
    
    #%% Wall thickness calculation
    t_wall_list = []
    for i in range(len(ContourCoordInt["x"])):
        t_wall = Px_list[i] * ContourCoordInt["y"][i] * Profile["SF_w"] * Profile["SF_w_burst"] / Profile["Sigma_y"]
        t_wall_list.append(t_wall)
        
   #print(t_wall_list)
    print("t_max",np.max(t_wall_list))
    print("t_min",np.min(t_wall_list))
    print("t_avg",np.average(t_wall_list))
    
    plt.title("mach number")
    plt.plot(ContourCoordInt["x"],Mx_list)
    plt.xlabel("Axial distance, 0 = throat [m]")
    plt.ylabel("Mach [/]")
    plt.show()
    plt.title("Pressure")
    plt.plot(ContourCoordInt["x"],Px_list)
    plt.xlabel("Axial distance, 0 = throat [m]")
    plt.ylabel("Pressure [Pa]")
    plt.show()
    plt.title("Temperature")
    plt.plot(ContourCoordInt["x"],Tx_list)
    plt.xlabel("Axial distance, 0 = throat [m]")
    plt.ylabel("Temperature [K]")
    plt.show()
    plt.title("Density")
    plt.plot(ContourCoordInt["x"],Rho_x_list)
    plt.xlabel("Axial distance, 0 = throat [m]")
    plt.ylabel("Density [kg/m³]")
    plt.show()
    
    return {"Prof":Profile,"Vc":V_c , "Vt":V_t , "Tt":T_t ,"vt":v_t , "ve":v_e , "mtot":m_tot , "At": At, "Dt":Dt ,"Ae": Ae, "De":De ,"Ac": Ac, "Dc":Dc,"Contour":ContourCoordInt,
            "Mx":Mx_list,"Px":Px_list,"Tx":Tx_list,"RhoX":Rho_x_list,"Dp":Dp,"Lg":Lg,"Lch":L_tot,"Dch":Dch,"Mch":M_ch}



#%% THERMAL Analysis

def THERM_ANSYS(Geo,CoolingType,material,t_Wall,t_annulus,plots):
    """
    This function determines the thermal loads for the given geometry.
    """
    Profile = Geo["Prof"]
    gamma_c =  CEA_RUN(Geo["Prof"])["gamma_c"]
    R_spec = CEA_RUN(Geo["Prof"])["Rspec"]
    visc_c = CEA_RUN(Geo["Prof"])["visc_c"]
    C_star = CEA_RUN(Geo["Prof"])["C*"]
    Pc = CEA_RUN(Geo["Prof"])["Pc"]
    m = Geo["mtot"]
    m_f = m/(((Geo["Prof"])["OF"]+1))
    Tc = CEA_RUN(Geo["Prof"])["Tc"]
    Dt = Geo["Dt"]                                  #throat diameter
    rc = 1.5*Dt                                     #curvature of throat inlet
    print("visc_c",visc_c)
    Cp_c = CEA_RUN(Geo["Prof"])["Cp_c"]
    print("Cp",Cp_c)
    Kc = CEA_RUN(Geo["Prof"])["Kc"]
    Tb = (Geo["Prof"])["T_f"]
    print("Kc",Kc)
    Pr = Cp_c*visc_c/Kc
    print("prandtl",Pr)
    Contour = Geo["Contour"]
    Rho_x = Geo["RhoX"]
    Mx = Geo["Mx"]
    Tx = Geo["Tx"]
    Px = Geo["Px"]
    Re_list =[]
    Darcy_friction =[]
    h_DB = []
    h_gnielinski = []
    h_SB = []
    h_MB = []
    T_r = []
    T_f = []
    
    if material == 'Cu':
        SpecHeat = 390                                 #Specific heat capacity in J/kgK
        RhoWall = 8960                                  #wall material density in kg/m³
        Tw = 0.6*1380                                        #max allowable wall temperature
    elif material == 'St':
        SpecHeat = 500                                  #Specific heat capacity in J/kgK
        RhoWall = 7850                                  #wall material density in kg/m³
        Tw = 0.6 * 1650                                       #max allowable wall temperature
    elif material == 'Al':
        SpecHeat = 880
        RhoWall = 2710
        Tw = 0.6*960                                        #max allowable wall temperature
    elif material == 'In718':
        SpecHeat = 435
        RhoWall = 2190
        Tw = 0.6*1573
            
    for i in range(len(Geo["Contour"]["x"])):
        Re_list.append(Rho_x[i]*(Contour["y"][i])*2*Mx[i]*np.sqrt(gamma_c * Tx[i] * R_spec) / visc_c)
        def f_coeff(f):
            return -2 * np.log10(0.15e-3/(3.7*Contour["y"][i]*2) + 2.51/(Re_list[i]*np.sqrt(f))) - 1/np.sqrt(f)
        fx = float(fsolve(f_coeff,0.01)[0])
        Darcy_friction.append(fx)
        #%%Dittus Boelter
        h_DB.append((Kc/(Contour["y"][i]*2))*0.023 * Re_list[i]**0.8 * Pr**0.33)
        #%% Gnielinski
        h_gnielinski.append((Kc/(Contour["y"][i]*2)) * (Darcy_friction[i]/8 * (Re_list[i]-1000)*Pr)/(1 + 12.7*(Darcy_friction[i]/8)**0.5 * (Pr**(2/3)-1)))
        #%% Bartz
        if Mx[i] <=1:
            Tr = Tx[i]*(1 + Pr**(1/3)*(gamma_c-1)/2 * Mx[i]**2)  #adiabatic wall temp
        elif Mx[i] >1:
            Tr = Tx[i]*(1 + Pr**(1/3)*(gamma_c-1)/2 * Mx[i]**2)
        Bl_corr = (1+ Mx[i]**2 * (gamma_c - 1)/2)**(-0.12) / (0.5 + 0.5*(Tw/Tx[i])*(1 + Mx[i]**2 *(gamma_c - 1)/2))**0.68
        #Tf = Tx[i]*((1+Tw/Tx[i])/2 + 0.22 *(gamma_c-1)/2 *Mx[i]**2)
        Tf = 0.5*Tw + 0.28*Tx[i]+0.22*Tr
        T_r.append(Tr)
        T_f.append(Tf)
        h_SB.append((0.026/(Contour["y"][i]*2)**0.2)*(visc_c**0.2 * Cp_c/Pr**0.6)*(Px[i]/C_star)**0.8 *(Dt/rc)**0.1*(Dt/(Contour["y"][i]*2))**1.8 * Bl_corr)
        h_MB.append(0.026*((m/(Contour["y"][i]**2 * np.pi))**0.8 /((Contour["y"][i]*2)**0.2) ) * (visc_c**0.2 * Cp_c/Pr**0.6) * (Tx[i]/Tf)**0.68)
        
    #Average of the maxima of the different heat transfer models:
    h_max_avg = np.average([np.max(h_DB),np.max(h_SB),np.max(h_MB),np.max(h_gnielinski)])
    Tf_avg = np.average(Tf)
    print("Tf_avg",Tf_avg)
    h_avg = []
    for i in range(len (Geo["Contour"]["x"])):
        h_avg.append( np.average([(h_DB[i]),(h_SB[i]),(h_MB[i]),(h_gnielinski[i])]))
    #the average value of the different models will be used for the rest of the thermal analysis
    if plots =="TRUE":
        #print(Re_list)
        #print(Darcy_friction)
        print("Avg of max heat transfer coeff", h_max_avg)
    
    
        plt.title("Re")
        plt.plot(Contour["x"],Re_list)
        plt.xlabel("Axial distance, 0 = throat [m]")
        plt.ylabel("Reynolds number [/]")
        plt.show()
    
        plt.title("Darcy friction factor")
        plt.plot(Contour["x"],Darcy_friction)
        plt.xlabel("Axial distance, 0 = throat [m]")
        plt.ylabel("Darcy friction factor [/]")
        plt.show()
    
        plt.title("heat_transfer coeff")
        plt.grid()
        plt.xlabel("Axial distance, 0 = throat [m]")
        plt.ylabel("Heat transfer coeff h [W/m²K]")
        plt.plot(Contour["x"],h_DB,label="Dittus-Boelter")
        plt.plot(Contour["x"],h_gnielinski,label="Gnielinski")
        plt.plot(Contour["x"],h_SB,label="Standard Bartz")
        plt.plot(Contour["x"],h_MB,label="Modified Bartz")
        plt.plot(Contour["x"],h_avg,label="Avg")
        plt.legend()
        plt.show()
    
          
    S_mantle = np.abs(np.trapz(2*Contour["y"]*np.pi,np.abs(Contour["x"])))
    print("S",S_mantle,"m²")
    V_mantle = S_mantle * t_Wall
    m_mantle = V_mantle * RhoWall
    print("m",m_mantle,"kg")
    if CoolingType == 'Sink':
        time = np.arange(0,Profile["tb"],0.05)
        T0 = Tc * (2/(gamma_c + 1))
        T_init = 288.15
        Twall = (T_init-Tf_avg)*1/np.exp(h_max_avg*S_mantle/(m_mantle*SpecHeat)*time) + Tf
        Twall2 = (T_init-Tf_avg)*1/np.exp(np.average(h_avg)*S_mantle/(m_mantle*SpecHeat)*time) + Tf
        Twall3 = (T_init-Tf_avg)*1/np.exp(np.min(h_avg)*S_mantle/(m_mantle*SpecHeat)*time) + Tf
        plt.title('Sink cooling temperature as function of burn time')
        plt.grid() 
        plt.xlabel("Time[s]")
        plt.ylabel("Wall temperature [K]")  
        plt.plot(time,Twall,label=f'{material} Twall, h_max')
        plt.plot(time,Twall2,label=f'{material} Twall, h_avg')
        plt.plot(time,Twall3,label=f'{material} Twall, h_min')
        plt.axhline(y = Tw, color = 'r', linestyle = '-') 
        plt.legend()
        plt.show()
    
    return {"Mw":m_mantle}

#%% PREDICTED HRE PERFORMANCE

def PERF_ANSYS(Profile, Geo, plot, output):
    steps = 100
    t_list = np.arange(0, Profile["tb"], Profile["tb"]/steps)  # creates a time array
    Dp = Geo["Dp"]
    Lg = Geo["Lg"]
    Pc = Profile["DesignChamberPressure"]*1e5

    OF = Profile["OF"]
    # m_ox = np.arange(Profile["m_o_dot"], 0.6*Profile["m_o_dot"], 0.6*Profile["m_o_dot"]/steps )
    # Define parameters
    m_dot_initial = 0.2 * Profile["m_o_dot"]  # Initial mass flow rate
    m_dot_nom = Profile["m_o_dot"]  # nominal mass flow rate
    m_dot_final = 0.8*Profile["m_o_dot"]  # Final mass flow rate
    ramp_up_time = 0.5  # Time for the ramp-up phase (in seconds)
    total_burn_time = Profile["tb"]  # Total burn time (in seconds)

    # Calculate time steps for ramp-up and steady-state phases
    ramp_up_steps = int(ramp_up_time / (total_burn_time / steps))
    steady_state_steps = steps - ramp_up_steps

    # Create time array for ramp-up phase
    t_ramp_up = np.linspace(0, ramp_up_time, ramp_up_steps)
    m_dot_ramp_up = m_dot_initial + (m_dot_nom - m_dot_initial) * t_ramp_up / ramp_up_time

    # Create time array for steady-state phase
    #t_steady_state = np.linspace(ramp_up_time, total_burn_time, steady_state_steps)
    m_dot_steady_state = np.linspace(m_dot_nom, m_dot_final, steady_state_steps)

    # Combine the two phases into a single array
    #m_ox = np.linspace(Profile["m_o_dot"], 0.8 * Profile["m_o_dot"], num=steps)
    m_ox = np.concatenate((m_dot_ramp_up, m_dot_steady_state))
    
    #print(m_ox)
    Pc_list =[]
    F_list = []
    M_f_list = []
    r_list = []
    OF_list = []
    Dp_list = []
    Isp_list = []
    G_list = []
    for i in range(len(t_list)):
        fuel = Fuel(Profile["Fuel"], temp=Profile["T_f"]) 
        ox = Oxidizer(Profile["Oxidizer"], temp=Profile["T_o"])
        Case = RocketProblem(pressure = Pc/1e5 ,pressure_units="bar", materials=[fuel, ox], o_f=OF,fac_ac = 10, pip= Pc/(Profile["Popt"])) 
        results = Case.run()
        gamma_t   = results.t_gamma
        gamma_c   = results.c_gamma
        Tc          = results.c_t
        Tt          = results.t_t
        Cp_e        = results.cp * 1e3
        Cp_c        = results.c_cp * 1e3
        Rho_c       = results.c_rho
        Rho_t       = results.t_rho
        R_spec      = (1/Rho_c)*(Pc)/Tc
        a_c         = results.c_son
        a_t         = results.t_son
        Cf          = results.cf
        Cstar       = results.cstar
        Mach_e      = results.mach
        ISP_e       = results.isp
        ISP_t       = results.t_isp
        ISP_vac     = results.ivac
        r = 1e-3 * Profile["a0"] * (4* m_ox[i]/(Dp**2 * np.pi))**Profile["k"]  #regression rate in [m/s]
        S = Dp * np.pi * Lg                                                 #exposed burning area [m²]
        mf = Profile["Rho_f"] * r * S
        def PR(x):
            return Geo["At"]/Geo["Ae"]-(((gamma_c +1)/2)**(1/(gamma_c -1)) * x**(1/gamma_c) * np.sqrt((gamma_c +1)/(gamma_c -1) * (1-x**((gamma_c -1)/gamma_c))))
        PR = fsolve(PR, 0.02 )[0]
        Pe = Pc*PR
        v_e = np.sqrt((2*gamma_c)/(gamma_c - 1)* R_spec * Tc*(1-(PR)**((gamma_c - 1)/gamma_c)))
        F = (mf + m_ox[i]) * v_e + Geo["Ae"]*(Pe - Profile["Popt"])
        OF = m_ox[i]/mf
        Pc = Cstar * Profile["Rho_f"]*r*S/Geo["At"]*(OF + 1)
        Mf = Profile["Rho_f"] * Lg * (Geo["Dc"]**2 - Dp**2) * np.pi/4
        if Dp < 0: 
            break
        Pc_list.append(Pc)
        M_f_list.append(Mf)
        OF_list.append(OF)
        Isp_list.append(ISP_e)
        Dp_list.append(Dp)
        F_list.append(F)
        r_list.append(r)
        G_list.append(4*m_ox[i]/(Dp**2 *np.pi))
        """print("OF",OF)
        print("r",r)
        print("mf",mf)
        print("Pc",Pc)
        print("Pe",Pe)
        print("F",F)
        print("Dp",Dp)"""
        Dp = Dp + 2*r*(Profile["tb"]/steps) #computes the new Dp, based on the current regression rate
        
    OF_avg = np.average(OF_list)
    Pc_avg = np.average(Pc_list)
    F_avg = np.average(F_list)
    print("OF_avg:",OF_avg)
    print("Pc_avg [Pa]:",Pc_avg)
    print("F_avg [N]:",F_avg)
       
    if plot == "TRUE":
        plt.title("Thrust vs time")
        plt.grid()
        plt.plot(t_list,F_list)
        plt.axhline(y = F_avg, color = 'r', linestyle = 'dashed', label = 'F_avg [N]')
        plt.xlabel("time [s]")
        plt.ylabel("Thrust [N]")
        plt.legend()
        plt.show()
    
        plt.title("Regression rate vs time")
        plt.grid()
        plt.plot(t_list,r_list)
        plt.xlabel("time [s]")
        plt.ylabel("Regression rate [m/s]")
        plt.show()
    
        plt.title("OF vs time")
        plt.grid()
        plt.plot(t_list,OF_list)
        plt.axhline(y = OF_avg, color = 'r', linestyle = 'dashed', label = 'OF_avg')
        plt.xlabel("time [s]")
        plt.ylabel("O/F [/]")
        plt.legend()
        plt.show()
    
        plt.title("M_Fuel vs time")
        plt.grid()
        plt.plot(t_list,M_f_list)
        plt.xlabel("time [s]")
        plt.ylabel("Fuel mass [kg]")
        plt.show()
    
        plt.title("Pc vs time")
        plt.grid()
        plt.plot(t_list,Pc_list)
        plt.axhline(y = Pc_avg, color = 'r', linestyle = 'dashed', label = 'Pc_avg [Pa]')
        plt.xlabel("time [s]")
        plt.ylabel("Pressure [Pa]")
        plt.legend()
        plt.show()
    
        plt.title("G_ox vs time")
        plt.grid()
        plt.plot(t_list,G_list)
        plt.xlabel("time [s]")
        plt.ylabel("Oxidizer mass flux [kg/m²s]")
        plt.show()
        
    if output == "CSV":
        # Create a Pandas DataFrame
        data = []
        data.append({'Oxidizer Mass Flow': Profile["m_o_dot"] ,
                'tank_diameter':Geo["Dch"] ,
                'tank_length':Geo["Lch"],
                'dry_mass':Geo["Mch"]+Profile["m_tank"],
                'dry_inertia':0,
                'nozzle_radius':Geo["Dt"]/2,
                'grain_number':1,
                'grain_separation':0,
                'grain_outer_radius':Geo["Dc"]/2,
                'grain_initial_inner_radius':Geo["Dp"]/2,
                'grain_initial_height':Geo["Lg"],
                'grain_density': Profile["Rho_f"],
                'grains_center_of_mass_position':Geo["Lg"]/2,
                'center_of_dry_mass_position':0.284,
                'nozzle_position':0,
                'burn_time':Profile["tb"],
                'throat_radius':Geo["Dt"]/2
        })
        
        
        ThrustProf= {
            'Thrust': F_list,
            'time': t_list
            }
        Rocket_data = pd.DataFrame(data)
        Thrust_curve = pd.DataFrame(ThrustProf)

        # Export the DataFrame to a CSV file
        Rocket_data.to_csv('rocket_data.csv',index = False)
        Thrust_curve.to_csv('ThrustCurve.csv',index = False)
    return



#%% Working Area

"""
Here all code related to the actual engine design can take place, everything above this section is simply the different 
functions.
"""
It_Req(HRE_1,20, 84,3000,0.7)
#HRE_1_Geo = CH_Geo(HRE_1, 'Bell', 10, 0.8 ,45, 35, 5,"OF","True")
#HRE_1_Geo = CH_Geo(HRE_1, 'Bell', 10, 1.5 ,45, 35, 2,"OF","True")
HRE_1_Geo = CH_Geo(HRE_1, 'Bell', 10, 1.5 ,45, 35, 2,"FreeCad","True")
#THERM_ANSYS(HRE_1_Geo, "Sink", "Cu", 10e-3,0,"TRUE")
#PERF_ANSYS(HRE_1, HRE_1_Geo, "TRUE", "CSV")