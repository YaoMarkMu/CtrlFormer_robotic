import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

prefix = '/apdcephfs/share_1290939/shoufachen/EXP_DR/2021.11.04'

# [0.0001, 0.00025, 0.0005, 0.0008, 0.001, 0.0025, 0.005]
lr_candidates = [0.0001, 0.0005, 0.001]

for lr in lr_candidates:
    print(lr)
    paths = glob.glob(os.path.join(prefix, r'*,env=cartpole_swingup,lr={},seed=*'.format(lr)))
    paths = [path + "/eval.csv" for path in paths]
    p = pd.DataFrame([])
    for i in range(len(paths)):
        try:
            p = pd.concat([p, pd.read_csv(paths[i])])
        except:
            print("Not existing:", paths[i])
    sns.lineplot(data=p, x="step", y="episode_reward", label="lr={}".format(lr))

# sns.get_figure().show()
plt.show()
