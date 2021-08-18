#%%
import wmtsa.modwt
import numpy as np
#%%
x=np.zeros(128)
x[40:46] = 0.5*np.cos(3*np.pi*np.arange(40,46)/16. + 0.08)
wa,sc = wmtsa.modwt.modwt (x,nlevels=4, boundary='periodic')
de,sm = wmtsa.modwt.imodwt_mra(wa, sc)
war,scr = wmtsa.modwt.cir_shift(wa,sc)
varw,vars = wmtsa.modwt.rot_cum_wav_svar(wa,sc)

# %%
