import pandas as pd
import openpyxl

def merge_ABCD():
    df1 = pd.read_csv('./AVI-RVI-data/ABCD_AVI_RVI.csv')
    df2 = pd.read_csv('./Saved-Model-Results/ABCD_XGB_tuned.csv')
    df3 = df1.merge(df2, how='outer', on='ID')
    df3.to_csv('./Merges/ABCD_AVI_RVI_XGB.csv', index=False)

def merge_HCP():
    df1 = pd.read_csv('./AVI-RVI-data/HCP_AVI_RVI.csv')
    df2 = pd.read_csv('./Saved-Model-Results/HCP_XGB_tuned.csv')
    df3 = df1.merge(df2, how='outer', on='ID')
    df3.to_csv('./Merges/HCP_AVI_RVI_XGB.csv', index=False)

def merge_ACP():
    df1 = pd.read_csv('./AVI-RVI-data/ACP_AVI_RVI.csv')
    df2 = pd.read_csv('./Saved-Model-Results/ACP_XGB_tuned.csv')
    df3 = df1.merge(df2, how='outer', on='ID')
    df3.to_csv('./Merges/ACP_AVI_RVI_XGB.csv', index=False)

merge_ABCD()
merge_HCP()
merge_ACP()