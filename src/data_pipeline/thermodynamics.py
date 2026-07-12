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
        
        # If columns exist, calculate features
        if all(col in df.columns for col in ['P2', 'Pamb']):
            df_phys['PR_comp'] = self.compute_compressor_pressure_ratio(df['P2'], df['Pamb'])
            
        if all(col in df.columns for col in ['P4', 'P3']):
            df_phys['PR_turb'] = self.compute_turbine_pressure_ratio(df['P4'], df['P3'])
            
        if all(col in df.columns for col in ['Tamb', 'T2', 'Pamb', 'P2']):
            df_phys['Comp_Isentropic_Efficiency'] = self.compute_isentropic_efficiency(df['Tamb'], df['T2'], df['Pamb'], df['P2'])
            
        if all(col in df.columns for col in ['T2', 'T3']):
            df_phys['Combustion_Temp_Rise'] = df['T3'] - df['T2']
            
        if all(col in df.columns for col in ['RPM', 'Tamb']):
            df_phys['Normalized_RPM'] = df['RPM'] / np.sqrt(df['Tamb'])
            
        return df_phys
