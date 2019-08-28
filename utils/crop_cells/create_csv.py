import glob, os
import pandas as pd

os.chdir("../../data/")
original_cwd = os.getcwd()

df = pd.read_csv("raw/train.csv")
records = df.to_records(index=False)
site1_ncells = []
site2_ncells = []

for index in range(len(records)):
    record = records[index]
    print(record)
    os.chdir(original_cwd)
    os.chdir(f'cropped/train/{record.experiment}/Plate{record.plate}/')
    
    site1_ncells.append(int(len(glob.glob(f'{record.well}_s1*.png'))/6))
    site2_ncells.append(int(len(glob.glob(f'{record.well}_s2*.png'))/6))

df['site1_ncells'] = site1_ncells
df['site2_ncells'] = site2_ncells


os.chdir(original_cwd)
df.to_csv("cropped/train.csv",index=False)
