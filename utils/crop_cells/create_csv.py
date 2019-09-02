import glob, os
import pandas as pd

os.chdir("../../data/")
original_cwd = os.getcwd()

"""
#do train csv
df = pd.read_csv("raw/train.csv")
"""
#do test csv
df = pd.read_csv("raw/test.csv")

records = df.to_records(index=False)
site1_ncells = []
site2_ncells = []

for index in range(len(records)):
    record = records[index]
    print(record)
    os.chdir(original_cwd)
    
    """
    # do train csv
    os.chdir(f'cropped/train/{record.experiment}/Plate{record.plate}/')
    """
    #do test csv
    os.chdir(f'cropped/test/{record.experiment}/Plate{record.plate}/')
    
    site1_ncells.append(int(len(glob.glob(f'{record.well}_s1*.png'))/6))
    site2_ncells.append(int(len(glob.glob(f'{record.well}_s2*.png'))/6))

df['site1_ncells'] = site1_ncells
df['site2_ncells'] = site2_ncells


os.chdir(original_cwd)
"""
#do train csv
df.to_csv("cropped/train.csv",index=False)
"""
#do test csv
df.to_csv("cropped/test.csv",index=False)
