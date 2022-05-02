import numpy as np   
import pandas as pd
from scipy.stats import skew
from scipy.signal import find_peaks

def create_features(chunk, periods=[25, 75, 125, 100, 175, 200, 225, 275, 300, 325]):
        # chunk['chunk_id'] = str(room)+'_'+str(device)+'_'+str(i)
        chunk['RSSI_diff'] = chunk.RSSI_Left - chunk.RSSI_Right
        chunk['RSSI_Left_diff'] = chunk.RSSI_Left.diff()
        chunk['RSSI_diff_diff'] = chunk.RSSI_diff.abs().diff()
        chunk['RSSI_Right_diff'] = chunk.RSSI_Right.diff()
        chunk['RSSI_rms'] = np.sqrt(chunk.RSSI_Left**2 + chunk.RSSI_Right**2)
        chunk['RSSI_rms_diff'] = chunk.RSSI_Right.diff()

        for col_name in [#'RSSI_Left', 'RSSI_Right', 
                         'RSSI_diff', 'RSSI_rms', 
                         'RSSI_Left_diff', 'RSSI_Right_diff', 'RSSI_rms_diff', 'RSSI_diff_diff']:
            for period in periods:
                feature_std = (chunk[col_name].rolling(period, min_periods=0).std()).fillna(0.0000000001)
                feature_mean = chunk[col_name].rolling(period, min_periods=0).mean()
                feature_max = chunk[col_name].rolling(period, min_periods=0).max()
                feature_min = chunk[col_name].rolling(period, min_periods=0).min()
                
                chunk[col_name+'_'+str(period)] = (chunk[col_name] - feature_mean)/feature_std
                chunk[col_name+'_'+str(period)+'_range'] = (feature_max - feature_min)/feature_std
                
                chunk[col_name+'_'+str(period)+'_skew'] = (feature_mean - (feature_max - feature_min)/2)/feature_std

        chunk.drop(['RSSI_Left', 'RSSI_Right', 'RSSI_diff', 'RSSI_rms', 
                         'RSSI_Left_diff', 'RSSI_Right_diff', 'RSSI_rms_diff', 'RSSI_diff_diff'], axis=1, inplace=True)
        
        return chunk


def create_minmax_features(chunk, periods=[10, 15, 25, 50, 100, 150, 200, 250, 300]):
        # chunk['chunk_id'] = str(room)+'_'+str(device)+'_'+str(i)
        chunk['RSSI_diff'] = chunk.RSSI_Left - chunk.RSSI_Right
        chunk['RSSI_Left_diff'] = chunk.RSSI_Left.diff()
        chunk['RSSI_diff_diff'] = chunk.RSSI_diff.abs().diff()
        chunk['RSSI_Right_diff'] = chunk.RSSI_Right.diff()
        chunk['RSSI_rms'] = np.sqrt(chunk.RSSI_Left**2 + chunk.RSSI_Right**2)
        chunk['RSSI_rms_diff'] = chunk.RSSI_Right.diff()

        for col_name in [#'RSSI_Left', 'RSSI_Right', 
                         'RSSI_diff', 'RSSI_rms', 
                         'RSSI_Left_diff', 'RSSI_Right_diff', 'RSSI_rms_diff', 'RSSI_diff_diff']:
            for period in periods:
                feature_std = (chunk[col_name].rolling(period, min_periods=0).std()).fillna(0.0000000001)
                feature_mean = chunk[col_name].rolling(period, min_periods=0).mean()
                feature_max = chunk[col_name].rolling(period, min_periods=0).max()
                feature_min = chunk[col_name].rolling(period, min_periods=0).min()
                
                chunk[col_name+'_'+str(period)] = (chunk[col_name] - feature_mean)/feature_std
                chunk[col_name+'_'+str(period)+'_range'] = (feature_max - feature_min)/feature_std
                chunk[col_name+'_'+str(period)+'_skew'] = (feature_mean - (feature_max - feature_min)/2)/feature_std

        chunk.drop(['RSSI_Left', 'RSSI_Right', 'RSSI_diff', 'RSSI_rms', 
                         'RSSI_Left_diff', 'RSSI_Right_diff', 'RSSI_rms_diff', 'RSSI_diff_diff'], axis=1, inplace=True)
        data = {}
        for col_name in chunk.columns[chunk.columns.str.startswith('RSSI')]:
                data[col_name+'_max'] = chunk[col_name].max()
                data[col_name+'_min'] = chunk[col_name].min()
        
        data = pd.DataFrame([data])
        
        return data