import numpy as np, pandas as pd

m1 = 'sub/lgbmod.csv'
m2 = 'sub/lgbmod2.csv'
m3 = 'sub/lgbmod2.csv'
m4 = 'sub/lgbmod2.csv'
m5 = 'sub/lgbmod2.csv'
m6 = 'sub/lgbmod2.csv'
m7 = 'sub/lgbmod2.csv'
m8 = 'sub/lgbmod2.csv'
# m9 = 'sub/nn.csv'
# m10 = 'sub/imp.csv'
# m11 = 'sub/ens_submission.csv'
# m12 = 'sub/ens_submission1.csv'
# m13 = 'sub/lgbmod_pseudo.csv'
m14 = 'sub/blended.csv'

m15 = 'sub/ens_submission3.csv'
m16 = 'sub/uplgb_pseudo_comp.csv'

f1 = pd.read_csv(m1)
f2 = pd.read_csv(m2)
f3 = pd.read_csv(m3)
f4 = pd.read_csv(m4)
f5 = pd.read_csv(m5)
f6 = pd.read_csv(m6)
f7 = pd.read_csv(m7)
f8 = pd.read_csv(m8)
# f9 = pd.read_csv(m9)
# f10 = pd.read_csv(m10)
# f11 = pd.read_csv(m11)
# f12 = pd.read_csv(m12)
# f13 = pd.read_csv(m13)
f14 = pd.read_csv(m14)

f15 = pd.read_csv(m15)
f16 = pd.read_csv(m16)

lc = ['project_is_approved']
p_res = f1.copy()
# p_res[lc] = f1[lc]*0.05+f2[lc]*0.05+f3[lc]*0.05+f4[lc]*0.05+f5[lc]*0.05+f6[lc]*0.05+f7[lc]*0.05+f8[lc]*0.05+f14[lc]*0.6
# p_res[lc] = f11[lc]*0.4+f9[lc]*0.3+f10[lc]*0.3
p_res[lc] = f15[lc]*0.7+f16[lc]*0.3
p_res.to_csv('sub/ens_submission4.csv', index=False)