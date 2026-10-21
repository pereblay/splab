"""Comprobaciones numéricas: ejecutar python test_plasma.py."""
import numpy as np
from scipy.integrate import trapezoid
from plasma_model import *
for kt in [1,50,300]:
    t=np.geomspace(kt*1e-9,kt*100,30000)
    assert abs(trapezoid(mj_pdf(t,kt),t)-1)<1e-6
rng=np.random.default_rng(13)
e=np.geomspace(.01,1000,10000); n=isotropic(len(e),rng); b=isotropic(len(e),rng)*.8
q,v=boost(e,n,b); e2,n2=boost(q,v,-b)
assert np.allclose(e2,e,rtol=1e-12) and np.allclose(n2,n,atol=1e-12)
assert np.allclose(kn_ratio(np.array([1e-8])),1,atol=1e-7)
# Electrones en reposo: recoil exacto y pérdida energética.
s,ns=scatter(e,n,np.zeros_like(n),rng)
mu=np.sum(n*ns,axis=1)
assert np.allclose(s,e/(1+e/MEC2*(1-mu)),rtol=1e-10)
assert np.all(s<=e*(1+1e-12))
z,c=transport(e,50,0,rng)
assert np.array_equal(z,e) and np.all(c==0)
z,c=transport(e,50,.7,rng)
assert len(z)==len(e) and np.all(np.isfinite(z)) and np.all(z>0)
energies=np.geomspace(1,100,30)
a=brems(energies,50,1e17,100,nquad=1500)
a2=brems(energies,50,1e17,100,nquad=3000)
assert np.all(a>0) and np.max(abs(a2/a-1))<.002
assert np.allclose(brems(energies,50,2e17,100),4*a)
assert np.allclose(brems(energies,50,1e17,200),8*a)
# Límite Born no relativista (Elwert tiende a 1 a Z*alpha/beta pequeño):
# comparar g_BH reconstruido sin Elwert con sqrt(3)/pi log((p1+p2)/(p1-p2)).
t=np.array([.5,1.,2.]); en=.1*t
p1=np.sqrt(t/MEC2*(2+t/MEC2)); p2=np.sqrt((t-en)/MEC2*(2+(t-en)/MEC2))
eta1=ALPHA*(1+t/MEC2)/p1; eta2=ALPHA*(1+(t-en)/MEC2)/p2
fe=eta2/eta1*np.expm1(-2*np.pi*eta1)/np.expm1(-2*np.pi*eta2)
g=bh_elwert(en,t)*p1**2*en/(2*ALPHA/np.sqrt(3)*SIGMA_T)/fe
gnr=np.sqrt(3)/np.pi*np.log((np.sqrt(t)+np.sqrt(t-en))/(np.sqrt(t)-np.sqrt(t-en)))
assert np.max(abs(g/gnr-1))<.02
print('OK: MJ, boosts, recoil, Thomson, tau=0, escape, escalados, cuadratura y límite NR.')
print('Cambio máximo al duplicar cuadratura:',np.max(abs(a2/a-1)))
