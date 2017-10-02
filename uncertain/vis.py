import pandas as pd
import matplotlib.pyplot as plt

f1 = pd.read_csv("rocketOrPPlant.csv")
f2 = pd.read_csv("rocketOrPPlantSmall.csv")
joined = pd.concat([f1, f2])
joined['numLabled'] = joined['data_dir'].str.extract('augmented_(\d+)')
joined['not_augmented'] = joined['data_dir'].str.contains('not_augmented')
joined['noprob'] = joined['data_dir'].str.contains('noprob')
noprob = joined[joined['noprob'] == True]
notaugmented = joined[joined['not_augmented'] == True]
augmentedWithProb = joined[(joined['noprob'] == False) & (joined['not_augmented'] == False)]

plt.figure();