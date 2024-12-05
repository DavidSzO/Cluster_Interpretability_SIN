
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import dask.dataframe as dd
import matplotlib.cm as cm
import os
import colorcet as cc

# Set paths
mainpath = 'C:/Users/david/OneDrive/Documents/FERV_documentos/RESULTS'
dic_cenarios = {
    'V1A1F2 2026': f'{mainpath}/V1A1F2_rev5/',
    'V1A1F3 2026': f'{mainpath}/V1A1F3_rev5/',
    'V1A1F4 2026': f'{mainpath}/V1A1F4_rev5/',
    'V1A1F5 2026': f'{mainpath}/V1A1F5_rev5/',
    'V2A2F2 2026': f'{mainpath}/V2A2F2_rev5/',
    'V2A2F3 2026': f'{mainpath}/V2A2F3_rev5/',
    'V2A2F4 2026': f'{mainpath}/V2A2F4_rev5/',
    'V2A2F5 2026': f'{mainpath}/V2A2F5_rev5/',
    'V3A3F2 2026': f'{mainpath}/V3A3F2_rev5/',
    'V3A3F3 2026': f'{mainpath}/V3A3F3_rev5/',
    'V3A3F4 2026': f'{mainpath}/V3A3F4_rev5/',
    'V3A3F5 2026': f'{mainpath}/V3A3F5_rev5/'
}

# Directory settings
dirGeral = 'StaticAnalysis/Data/Geral/'
dirIndice = 'StaticAnalysis/Data/Indice_n_2/'
dirRamos = 'StaticAnalysis/Data/Fluxo em Ramos/'
dirPotencia = 'StaticAnalysis/Data/Potencia/'

# File names
filenames = [
    (dirGeral + 'OPF_NC.csv', 'OPF_NC'), 
    (dirGeral + 'PWF_NC.csv', 'PWF_NC'),
    (dirGeral + 'Df_ger.csv', 'ger'), 
    (dirGeral + 'Df_nt.csv', 'nt'), 
    (dirRamos + 'Df_Linhas.csv', 'linhas'), 
    (dirRamos + 'Df_Trafo.csv', 'Trafo'), 
    (dirRamos + 'DF_Intercambios.csv', 'intercambios'), 
    (dirRamos + 'DF_HVDC.csv', 'HVDC'),
    (dirPotencia +'Df_Reserva_PO_MW.csv', 'pot_mw_reserv'),  
    (dirIndice + 'Df_DPI_S2.csv', 'Index'), 
    (dirIndice +'Df_DPI_S4.csv', 'Index_PO'), 
    (dirIndice + 'Df_DPI_S3.csv', 'Index_Modif'), 
    (dirIndice +'Df_PQ_DPI_S1.csv', 'IndexDec_PQ'), 
    (dirIndice +'Df_PV_DPI_S1.csv', 'IndexDec_PV'), 
    (dirPotencia+'Df_MW-MVAR_PO.csv', 'pot_mw_mvar'), 
    (dirPotencia+'DF_POT_Reg.csv', 'ger_reg')
]

# Define color palette
intense_palette = sns.color_palette(cc.glasbey, n_colors=12)
namescenarios = {key:[key,intense_palette[idx]] for idx, key in enumerate(dic_cenarios.keys())}
itemsforanalysis = [var for _,var in filenames]

# Optimized read_and_append function using Dask
def read_and_append(filename, cenario, lst):
    df = dd.read_csv(filename)  # Keep as Dask DataFrame for now
    df['Cenario'] = cenario
    # Avoid converting to pandas immediately; do this after processing
    lst.append(df)
    return lst

# Function to load and process all datasets
def load_and_process_data(dic_cenarios, filenames):
    keys = [i for _, i in filenames]
    dfs = {key: [] for key in keys}

    # Read and append data from each scenario and filename
    for cenario, path in dic_cenarios.items():
        for filename, key in filenames:
            dfs[key] = read_and_append(path + filename, cenario, dfs[key])

    # Convert all dask DataFrames to pandas after appending
    dfs = {key: dd.concat(lst).compute() for key, lst in dfs.items()}
    
    return dfs

# Optimized IndiceLinhas function with memory and performance improvements
def IndiceLinhas(df, n):
    # Reduce memory usage by filtering early and using efficient data types
    df['MW_Flow'] = np.where(df['MW:From-To'] >= 0, df['MW:From-To'], df['MW:To-From']).astype('float32')
    df['PI_mva'] = (df['% L1'] / 100) ** (2 * n)

    # Group by columns and aggregate after filtering unnecessary data
    df_r_nt_1 = df[df['VBASEKV'] >= 500].groupby(
        ['key', 'Cenario', 'REG']
    ).agg({'MW_Flow': 'sum', 'MVA': 'sum', 'PI_mva': 'sum'}).astype('float32')

    df_r_nt_2 = df[df['VBASEKV'] < 500].groupby(
        ['key', 'Cenario', 'REG']
    ).agg({'MW_Flow': 'sum', 'MVA': 'sum', 'PI_mva': 'sum'}).astype('float32')

    # Convert the PI_mva column after grouping to reduce operations on large DataFrames
    df_r_nt_1['PI_mva_500up'] = df_r_nt_1['PI_mva'] ** (1 / (2 * n))
    df_r_nt_2['PI_mva_500down'] = df_r_nt_2['PI_mva'] ** (1 / (2 * n))

    # Merge results and fill missing values
    df_r_nt = df_r_nt_1[['PI_mva_500up']].merge(df_r_nt_2[['PI_mva_500down']], on=['key', 'Cenario', 'REG'], how='outer')
    df_r_nt.fillna({'PI_mva_500up': 0, 'PI_mva_500down': 0}, inplace=True)

    return df_r_nt

# Optimized flatdf function
def flatdf(df, BG):
    df_reordered = df.reorder_levels(['key', 'Dia', 'Hora', 'Cenario', 'REG', BG])
    regionmap = {
        'Nordeste': 'Northeast', 'Norte': 'North', 'Sudeste-Centro-Oeste': 'SE-CW',
        'Sul': 'South', 'AC-RO': 'AC-RO'
    }
    df_reordered = df_reordered.rename(index=regionmap, level='REG')
    df_reordered_sorted = df_reordered.sort_index(level=['key', 'Dia', 'Hora', 'Cenario', 'REG'])
    df_reordered_sorted.rename(columns={'CSI_INF': 'DPI_inf', 'CSI_SUP': 'DPI_sup'}, inplace=True)

    return df_reordered_sorted

# Main execution flow (you can extend this as needed for further processing)

if __name__ == "__main__":
    print("Loading and processing data...")
    data = load_and_process_data(dic_cenarios, filenames)
    print("Data loaded and processed.")
