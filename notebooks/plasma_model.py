"""Plasma de H ionizado: BH-Elwert y transporte Compton relativista. CGS/keV."""
import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.special import kve
MEC2=510.99895
SIGMA_T=6.6524587321e-25
C=2.99792458e10
ALPHA=1/137.035999084
KEV=1.602176634e-9
KPC=3.085677581491367e21

def mj_pdf(T, kT):
    """Densidad normalizada por keV de energía cinética."""
    T=np.asarray(T); g=1+T/MEC2; p=np.sqrt((T/MEC2)*(2+T/MEC2))
    return g*p*np.exp(-T/kT)/(kT*kve(2,MEC2/kT))

def electron_grid(kT, custom=None):
    if not 1 <= kT <= 300: raise ValueError('kT debe estar entre 1 y 300 keV.')
    t=np.geomspace(kT*1e-8, max(60*kT,1000), 8000)
    f=mj_pdf(t,kT) if custom is None else np.asarray(custom(t),float)
    if f.shape!=t.shape or not np.all(np.isfinite(f)) or np.any(f<0):
        raise ValueError('La distribución debe ser finita, no negativa y del mismo tamaño que T.')
    norm=trapezoid(f,t)
    if norm<=0: raise ValueError('Distribución vacía en la malla electrónica.')
    f=f/norm
    cdf=cumulative_trapezoid(f,t,initial=0); cdf/=cdf[-1]
    return t,f,cdf

def bh_elwert(E,T):
    """d sigma/dE [cm²/keV], Z=1. Chluba+2020 ecs. 1,7,9."""
    T,E=np.broadcast_arrays(np.asarray(T,dtype=np.longdouble),np.asarray(E,dtype=np.longdouble))
    out=np.zeros(T.shape,dtype=np.longdouble); ok=T>E
    a=T[ok]/MEC2; w=E[ok]/MEC2; b=a-w
    g1=1+a; g2=1+b; p1=np.sqrt(a*(a+2)); p2=np.sqrt(b*(b+2))
    l1=np.arcsinh(p1); l2=np.arcsinh(p2)
    L=np.log((a+b+a*b+p1*p2)/w)
    g=(p1*p2/4-3*g1*g2/8*(p1/p2+p2/p1)+g1*g2*L
       +3*w*L/8*((1+g1*g2/p1**2)*l1/p1-(1+g1*g2/p2**2)*l2/p2
       +w*(1+(g1*g2+g1**2*g2**2)/(p1*p2)**2))
       +3/8*(g2*p2/p1**2*l1+g1*p1/p2**2*l2-2*l1*l2))*np.sqrt(3)/np.pi
    eta1=ALPHA*g1/p1; eta2=ALPHA*g2/p2
    elwert=eta2/eta1*np.expm1(-2*np.pi*eta1)/np.expm1(-2*np.pi*eta2)
    if np.any(g< -1e-8): raise ArithmeticError('Cancelación numérica en BH.')
    out[ok]=2*ALPHA/np.sqrt(3)*SIGMA_T/(p1*p1*w)*np.maximum(g,0)*elwert/MEC2
    return np.asarray(out,float)

def brems(E,kT,ne,radius_km,custom=None,nquad=1500):
    """Luminosidad espectral de fotones [s^-1 keV^-1], n_i=n_e."""
    grid,f,_=electron_grid(kT,custom)
    result=[]
    for e in np.asarray(E):
        if e>=grid[-1]: result.append(0.); continue
        # Malla desplazada: resuelve el umbral cinemático T=E.
        t=e+np.geomspace(max(1e-7,e*1e-8),grid[-1]-e,nquad)
        pdf=mj_pdf(t,kT) if custom is None else np.interp(t,grid,f,left=0,right=0)
        p=np.sqrt((t/MEC2)*(2+t/MEC2)); beta=p/(1+t/MEC2)
        result.append(trapezoid(pdf*C*beta*bh_elwert(e,t),t))
    return np.array(result)*ne**2*(4*np.pi/3)*(radius_km*1e5)**3

def isotropic(n,rng):
    v=rng.normal(size=(n,3)); return v/np.linalg.norm(v,axis=1)[:,None]

def boost(E,n,b):
    """Transforma fotón al marco que se mueve con velocidad b*c."""
    b2=np.sum(b*b,axis=1); gamma=1/np.sqrt(1-b2); dot=np.sum(b*n,axis=1)
    energy=gamma*E*(1-dot)
    coeff=np.divide(gamma-1,b2,out=np.zeros_like(b2),where=b2>0)*dot-gamma
    direction=(n+coeff[:,None]*b)/(gamma*(1-dot))[:,None]
    return energy,direction

def kn_ratio(x):
    x=np.asarray(x); y=np.maximum(x,1e-3); l=np.log1p(2*y)
    value=.75*((1+y)/y**3*(2*y*(1+y)/(1+2*y)-l)+l/(2*y)-(1+3*y)/(1+2*y)**2)
    return np.where(x<1e-3,1-2*x+5.2*x*x,value)

def scatter(E,n,b,rng):
    er,nr=boost(E,n,b); x=er/MEC2
    mu=np.empty(len(E)); pending=np.arange(len(E))
    while len(pending):
        trial=rng.uniform(-1,1,len(pending)); r=1/(1+x[pending]*(1-trial))
        accept=rng.random(len(pending)) < .5*r*r*(r+1/r-(1-trial*trial))
        mu[pending[accept]]=trial[accept]; pending=pending[~accept]
    axis=np.zeros_like(nr); axis[:,2]=1; axis[np.abs(nr[:,2])>.9]=[1,0,0]
    u=np.cross(nr,axis); u/=np.linalg.norm(u,axis=1)[:,None]; v=np.cross(nr,u)
    phi=rng.uniform(0,2*np.pi,len(E))
    nr2=mu[:,None]*nr+np.sqrt(1-mu*mu)[:,None]*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v)
    return boost(er/(1+x*(1-mu)),nr2,-b)

def transport(E,kT,tau,rng,source='beam',custom=None):
    """Esfera de radio unidad. Colisiones nulas con cota 2*n_e*sigma_T.
    Salida integrada sobre todas las direcciones; tau=n_e*sigma_T*R.
    """
    if not 0<=tau<=3: raise ValueError('Usar 0 <= tau radial <= 3.')
    E=np.array(E,float,copy=True); nphot=len(E); counts=np.zeros(nphot,int)
    if tau==0: return E,counts
    t,_,cdf=electron_grid(kT,custom)
    if source=='beam':
        r=np.sqrt(rng.random(nphot)); phi=rng.uniform(0,2*np.pi,nphot)
        pos=np.column_stack((r*np.cos(phi),r*np.sin(phi),-np.sqrt(1-r*r)))
        direction=np.tile([0.,0.,1.],(nphot,1))
    elif source=='internal':
        pos=isotropic(nphot,rng)*rng.random(nphot)[:,None]**(1/3)
        direction=isotropic(nphot,rng)
    else: raise ValueError('source: beam o internal')
    active=np.arange(nphot)
    for _ in range(20000):
        if not len(active): return E,counts
        dot=np.sum(pos[active]*direction[active],axis=1)
        edge=-dot+np.sqrt(np.maximum(0,dot*dot+1-np.sum(pos[active]**2,axis=1)))
        step=rng.exponential(1/(2*tau),len(active))
        active=active[step<edge]; step=step[step<edge]
        pos[active]+=step[:,None]*direction[active]
        g=1+np.interp(rng.random(len(active)),cdf,t)/MEC2
        b=isotropic(len(active),rng)*np.sqrt(1-1/g**2)[:,None]
        relative=1-np.sum(b*direction[active],axis=1)
        x=g*E[active]*relative/MEC2
        accepted=rng.random(len(active)) < relative*kn_ratio(x)/2
        ids=active[accepted]
        E[ids],direction[ids]=scatter(E[ids],direction[ids],b[accepted],rng)
        counts[ids]+=1
    raise RuntimeError('Transporte sin completar; no se devuelve un espectro truncado.')

def sample_spectrum(grid,photons,n,rng):
    cdf=cumulative_trapezoid(photons,grid,initial=0); total=cdf[-1]
    if total<=0: raise ValueError('Espectro semilla vacío.')
    return np.interp(rng.random(n)*total,cdf,grid),total

def histogram_flux(E,total_rate,bins,distance_kpc,n_original):
    """Flujo energético bin-promediado, con error Monte Carlo [erg/cm²/s/keV]."""
    weights=E*KEV*total_rate/n_original/(4*np.pi*(distance_kpc*KPC)**2)
    flux=np.histogram(E,bins,weights=weights)[0]/np.diff(bins)
    error=np.sqrt(np.histogram(E,bins,weights=weights**2)[0])/np.diff(bins)
    return flux,error
