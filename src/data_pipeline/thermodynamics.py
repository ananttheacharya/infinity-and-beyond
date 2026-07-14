import pandas as pd
import numpy as np

class ThermodynamicsEngine:
    """
    The Physics-Guided Data Pipeline. 
    This engine converts raw sensor data into thermodynamic invariants.
    """
    def __init__(self):
        # Specific heat ratio for air
        self.gamma = 1.4 
        # Gas constant for air J/(kg·K)
        self.R = 287.05 
        # Lower Heating Value of Jet-A Fuel (J/kg)
        self.LHV = 42.8e6

    def compute_compressor_pressure_ratio(self, p_out, p_in):
        """ PR_comp = P2 / Pamb """
        return p_out / p_in

    def compute_turbine_pressure_ratio(self, p_out, p_in):
        """ PR_turb = P4 / P3 """
        return p_out / p_in
        
    def compute_temperature_ratio(self, t_out, t_in):
        return t_out / t_in
        
    def compute_isentropic_efficiency(self, t_in, t_out, p_in, p_out):
        """
        Calculates Isentropic Efficiency for a compressor.
        eta_c = ( (P_out/P_in)^((gamma-1)/gamma) - 1 ) / ( (T_out/T_in) - 1 )
        """
        pr = self.compute_compressor_pressure_ratio(p_out, p_in)
        tr = self.compute_temperature_ratio(t_out, t_in)
        
        # Protect against division by zero or invalid values
        epsilon = 1e-6
        ideal_work = np.power(pr, (self.gamma - 1) / self.gamma) - 1
        actual_work = tr - 1
        
        efficiency = ideal_work / (actual_work + epsilon)
        return efficiency

    def extract_physics_features(self, df):
        """
        Transforms raw sensor DataFrame into physically meaningful features.
        Expected columns: 'Tamb', 'Pamb', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'RPM', 'Fuel_Flow'
        """
        df_phys = pd.DataFrame()
        
        # Original 5 valid features
        if all(col in df.columns for col in ['P2', 'Pamb']):
            df_phys['PR_comp'] = self.compute_compressor_pressure_ratio(df['P2'], df['Pamb'])
            
        if all(col in df.columns for col in ['P4', 'P3']):
            df_phys['PR_turb'] = self.compute_turbine_pressure_ratio(df['P4'], df['P3'])
            
        if all(col in df.columns for col in ['Tamb', 'T2', 'Pamb', 'P2']):
            df_phys['Comp_Isentropic_Efficiency'] = self.compute_isentropic_efficiency(df['Tamb'], df['T2'], df['Pamb'], df['P2'])
            
        # Combustion_Temp_Rise is perfectly collinear with Combustor_Heat_Addition, dropped.
        if all(col in df.columns for col in ['RPM', 'Tamb']):
            df_phys['Normalized_RPM'] = df['RPM'] / np.sqrt(df['Tamb'])
            
        # New valid features
        if all(col in df.columns for col in ['T4', 'T3', 'P4', 'P3']):
            # (1 - T4/T3) / (1 - (P4/P3)^0.2857)
            t_ratio = df['T4'] / df['T3']
            p_ratio = df['P4'] / df['P3']
            df_phys['Turb_Isentropic_Efficiency'] = (1 - t_ratio) / (1 - np.power(p_ratio, 0.2857) + 1e-6)
            
        cp = 1005.0 # J/(kg*K)
        if all(col in df.columns for col in ['T2', 'Tamb']):
            df_phys['Compressor_Specific_Work'] = cp * (df['T2'] - df['Tamb'])
            
        if all(col in df.columns for col in ['T3', 'T4']):
            df_phys['Turbine_Specific_Work'] = cp * (df['T3'] - df['T4'])
            
        if 'Turbine_Specific_Work' in df_phys.columns and 'Compressor_Specific_Work' in df_phys.columns:
            df_phys['Net_Specific_Work'] = df_phys['Turbine_Specific_Work'] - df_phys['Compressor_Specific_Work']
            
        if all(col in df.columns for col in ['T3', 'T2']):
            df_phys['Combustor_Heat_Addition'] = cp * (df['T3'] - df['T2'])
            
        # Overall_Pressure_Ratio is perfectly collinear with PR_comp, dropped.
            
        # LHV Energy Balance & Estimated Air Mass Flow
        # Assumes complete combustion and constant Cp. Note: T3 is uncharacteristically high (>3000K) 
        # in the raw data simulation, meaning real Cp would diverge significantly from 1005 J/kgK.
        if all(col in df.columns for col in ['Fuel_Flow', 'T2', 'T3']):
            # Q_in (total) = m_dot_fuel * LHV
            # m_dot_air = Q_in / (Cp * Delta T)
            q_in_total = df['Fuel_Flow'] * self.LHV
            delta_t_comb = df['T3'] - df['T2']
            df_phys['Estimated_Air_Mass_Flow'] = q_in_total / (cp * delta_t_comb + 1e-6)
            
            if 'Net_Specific_Work' in df_phys.columns:
                # Thermal efficiency = Net Work / Heat Added
                # Net Work = m_dot_air * Net_Specific_Work
                net_work_total = df_phys['Estimated_Air_Mass_Flow'] * df_phys['Net_Specific_Work']
                df_phys['Estimated_Thermal_Efficiency'] = net_work_total / (q_in_total + 1e-6)
            
        # Drop perfectly collinear and intermediate features to prevent small-sample memorization
        cols_to_drop = ['Combustion_Temp_Rise', 'Overall_Pressure_Ratio', 'Compressor_Specific_Work', 'Turbine_Specific_Work', 'Combustor_Heat_Addition']
        df_phys = df_phys.drop(columns=cols_to_drop, errors='ignore')
            
        return df_phys
