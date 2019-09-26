import pandas as pd
import numpy as np

def main():
    train = pd.read_csv("train_old.csv")
    if "group" not in train.columns: 
        #use an experiment that has all siRNAs
        exp = "HEPG2-03"
        train_exp = train.groupby("experiment").get_group(exp)
        
        #add group columnn to dataframe
        train["group"] = 0
        
        #get lists of sirna by group
        p1 = train_exp.groupby("plate").get_group(1)
        s1 = list(p1["sirna"])
        p2 = train_exp.groupby("plate").get_group(2)
        s2 = list(p2["sirna"])
        p3 = train_exp.groupby("plate").get_group(3)
        s3 = list(p3["sirna"])
        p4 = train_exp.groupby("plate").get_group(4)
        s4 = list(p4["sirna"])
        
        #get indices of rows corresponding to each group
        idx1 = [x in s1 for x in train["sirna"]]
        idx2 = [x in s2 for x in train["sirna"]]
        idx3 = [x in s3 for x in train["sirna"]]
        idx4 = [x in s4 for x in train["sirna"]]

        #assign group labels based on the indices
        train.loc[idx1, "group"] = 1
        train.loc[idx2, "group"] = 2
        train.loc[idx3, "group"] = 3
        train.loc[idx4, "group"] = 4

        #write new csv
        train.to_csv(path_or_buf="train.csv", index=False)
    
    if "group_target" not in train.columns: 
        #use an experiment that has all siRNAs
        exp = "HEPG2-03"
        train_exp = train.groupby("experiment").get_group(exp)
        
        #add group columnn to dataframe
        train["group_target"] = 0
        
        #get lists of sirna by group
        p1 = train_exp.groupby("plate").get_group(1)
        s1 = list(p1["sirna"])
        gt1 = np.argsort(np.argsort(s1))
        p2 = train_exp.groupby("plate").get_group(2)
        s2 = list(p2["sirna"])
        gt2 = np.argsort(np.argsort(s2))
        p3 = train_exp.groupby("plate").get_group(3)
        s3 = list(p3["sirna"])
        gt3 = np.argsort(np.argsort(s3))
        p4 = train_exp.groupby("plate").get_group(4)
        s4 = list(p4["sirna"])
        gt4 = np.argsort(np.argsort(s4))
        
        ordered_gt = np.zeros(1108)
        for idx, each in enumerate(s1):
            ordered_gt[each] = gt1[idx]
        for idx, each in enumerate(s2):
            ordered_gt[each] = gt1[idx]
        for idx, each in enumerate(s3):
            ordered_gt[each] = gt1[idx]
        for idx, each in enumerate(s4):
            ordered_gt[each] = gt1[idx]
        
        #for idx, each in enumerate(train['sirna']):
        #    train.loc[idx, "group_target"] = ordered_gt[each]
        for i in range(1108):
            train.loc[train["sirna"]==i, "group_target"] = int(ordered_gt[i])

        #write new csv
        train.to_csv(path_or_buf="train.csv", index=False)



if __name__ == '__main__':
    main()
    

