"""Comprobaciones físicas y numéricas; ejecutar python test_ciclosincrotron.py."""
from ciclosincrotron_model import *

# Normalización térmica desde régimen frío a ultrarrelativista.
for kt in [1,20,100,5000]:
    t=np.geomspace(kt/MEC2*1e-8,kt/MEC2*60,40000)
    assert abs(trapezoid(gamma_pdf(1+t,kt),t)-1)<1e-6

# Potencia total clásica de Liénard, electrones isotrópicos monoenergéticos.
beta=.1; g=1/np.sqrt(1-beta**2); kinetic=(g-1)*MEC2
mu,w=leggauss(24); mu=(mu+1)/2; w=w/2
x=np.geomspace(.85,4.1,120000); power=0
for u,weight in zip(mu,w):
    angle=np.degrees(np.arccos(u))
    spectrum=sum(harmonic_spectrum(x,n,angle,20,'monoenergetica',kinetic) for n in range(1,5))
    power+=4*np.pi*weight*trapezoid(spectrum,x)*NU_B_PER_G
expected=4*E_CHARGE**4/(9*M_E**2*C**3)*g*g*beta*beta
relative=abs(power/expected-1)
assert relative<.004,(power,expected,relative)
print(f'Potencia total vs Liénard: error relativo {relative:.3g}')

# Kernel sincrotrón: integral conocida 8*pi/(9*sqrt(3)).
x=np.geomspace(1e-9,1000,40000)
assert abs(trapezoid(synchrotron_F(x),x)/(8*np.pi/(9*np.sqrt(3)))-1)<1e-5

# Resolución instrumental conserva potencia; campo lognormal escala potencia con <B²>.
lognu=np.linspace(-10,10,20000); dx=lognu[1]-lognu[0]
y=np.exp(-lognu**2/2)
p0=trapezoid(y,lognu)
assert abs(trapezoid(broaden_log(y,dx,0,.2),lognu)/p0-1)<1e-6
assert abs(trapezoid(broaden_log(y,dx,.2,0),lognu)/p0-np.exp(.08))<1e-5

# A igual nu/B la potencia logarítmica escala como B².
a=compute_band(100,1e7,3e10,75,20,'termica',50,nmax=30,points=700)
b=compute_band(200,2e7,6e10,75,20,'termica',50,nmax=30,points=700)
assert np.allclose(b['total'],4*a['total'],rtol=1e-8,atol=1e-40)

# Convergencia: 80 -> 160 nodos para los primeros 10 órdenes.
x=np.geomspace(.3,15,900)
s80=sum(harmonic_spectrum(x,n,75,20,nquad=80) for n in range(1,11))
s160=sum(harmonic_spectrum(x,n,75,20,nquad=160) for n in range(1,11))
mask=s160>max(s160)*1e-7
error=np.max(np.abs(s80[mask]/s160[mask]-1))
assert error<.005,error
print(f'Cuadratura 80 -> 160: diferencia máxima significativa {error:.3g}')
print('OK: MJ, Liénard, kernel sincrotrón, convoluciones, escalado B y cuadratura.')
