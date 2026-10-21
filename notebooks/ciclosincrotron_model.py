"""Emisión clásica ciclotrón/sincrotrón, ópticamente delgada, electrones isotrópicos.
CGS, frecuencia en Hz. Salida por electrón y por estereorradián.
Referencia: Mahadevan, Narayan & Yi 1996, ecuaciones 2–8.
"""
import numpy as np
from scipy.special import jv, jvp, kve, kv
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.legendre import leggauss
from functools import lru_cache

E_CHARGE=4.80320471257e-10
M_E=9.1093837139e-28
C=2.99792458e10
H=6.62607015e-27
KEV=1.602176634e-9
MEC2=510.99895
B_Q=4.414005e13
NU_B_PER_G=E_CHARGE/(2*np.pi*M_E*C)

def gyrofrequency(B):
    return NU_B_PER_G*B

def gamma_pdf(gamma,kT):
    """Maxwell–Jüttner normalizada por unidad de gamma."""
    theta=kT/MEC2
    gamma=np.asarray(gamma)
    p=np.sqrt(np.maximum(0,(gamma-1)*(gamma+1)))
    return gamma*p*np.exp(-(gamma-1)/theta)/(theta*kve(2,1/theta))

def electron_limits(kT,distribution,kinetic_keV):
    if distribution=='termica':
        # Corte de la cola exponencial a T=40 kT.
        return 1+40*kT/MEC2
    if distribution=='monoenergetica': return 1+kinetic_keV/MEC2
    raise ValueError('Distribución: termica o monoenergetica.')

@lru_cache(maxsize=16)
def quadrature(order):
    return leggauss(order)

def harmonic_spectrum(x,order,theta_deg,kT=20.,distribution='termica',
                      kinetic_keV=50.,nquad=80):
    """dP_n/(dnu dOmega) / B; x=nu/nu_B.

    Integra analíticamente la delta de resonancia respecto a mu=cos(pitch).
    La distribución de pitch es isotrópica, dmu/2. Para theta != 90 grados:
    mu=(1-n/(gamma*x))/(beta*cos(theta)).
    El prefactor tras integrar la delta es pi*e²*nu/(c*beta*|cos(theta)|).
    """
    x=np.asarray(x,float)
    angle=np.deg2rad(theta_deg); cs=np.cos(angle); sn=np.sin(angle)
    if not 0<theta_deg<90: raise ValueError('Ángulo matemático: 0 < theta < 90 grados.')
    if np.any(x<=0): raise ValueError('Frecuencias positivas requeridas.')
    if int(order)!=order or order<1: raise ValueError('Orden armónico entero >= 1.')
    a=order/x
    constant=np.pi*E_CHARGE**2*NU_B_PER_G/C
    def angular(g,xx):
        beta=np.sqrt((g-1)*(g+1))/g
        mu=(1-order/(g*xx))/(beta*cs)
        valid=np.abs(mu)<=1
        mu=np.clip(mu,-1,1)
        bpar=beta*mu; bperp=beta*np.sqrt(np.maximum(0,1-mu*mu))
        D=1-bpar*cs
        arg=order*bperp*sn/D
        bracket=((cs-bpar)/sn)**2*jv(order,arg)**2+bperp**2*jvp(order,arg)**2
        return np.where(valid,bracket/beta,0)
    if distribution=='monoenergetica':
        g=1+kinetic_keV/MEC2
        return constant*x/abs(cs)*angular(g,x)
    if distribution!='termica': raise ValueError('Distribución desconocida.')
    # Soporte exacto de la resonancia para gamma: raíces de
    # sin²(theta)*gamma² - 2*a*gamma + a² + cos²(theta) <= 0.
    discriminant=a*a-sn*sn
    root=np.sqrt(np.maximum(0,discriminant))
    # Raíz pequeña racionalizada para evitar cancelaciones.
    lo=(a*a+cs*cs)/(a+abs(cs)*root)
    hi=(a+abs(cs)*root)/(sn*sn)
    upper=1+40*kT/MEC2
    lo=np.maximum(lo,1+1e-12); hi=np.minimum(hi,upper)
    valid=(discriminant>0)&(hi>lo)
    result=np.zeros_like(x)
    if not np.any(valid): return result
    # Integración logarítmica de gamma-1, adecuada también para plasmas fríos.
    z,w=quadrature(nquad)
    low=np.log(lo[valid]-1); high=np.log(hi[valid]-1)
    logt=low[:,None]+(z[None,:]+1)/2*(high-low)[:,None]
    t=np.exp(logt); g=1+t
    integrand=gamma_pdf(g,kT)*angular(g,x[valid,None])*t
    integral=np.sum(integrand*w,axis=1)*(high-low)/2
    result[valid]=constant*x[valid]/abs(cs)*integral
    return result

@lru_cache(maxsize=1)
def synchrotron_table():
    x=np.geomspace(1e-7,700,30000)
    # Integrar desde el extremo alto evita restar integrales casi iguales.
    integral=-cumulative_trapezoid(kv(5/3,x)[::-1],x[::-1],initial=0)[::-1]
    f=x*integral
    return x,f

def synchrotron_F(x):
    x=np.asarray(x,float); table,f=synchrotron_table()
    ans=np.exp(np.interp(np.log(np.maximum(x,1e-300)),np.log(table),
                        np.log(np.maximum(f,1e-300))))
    ans=np.where(x<table[0],2.1495282415344786*np.cbrt(x),ans)
    return np.where(x>650,np.sqrt(np.pi*x/2)*np.exp(-x),ans)

def continuum_spectrum(x,theta_deg,kT=5000.,distribution='termica',
                       kinetic_keV=10000.,nquad=200):
    """dP/(dnu dOmega)/B, aproximación ultrarrelativista de beaming.
    Para electrones isotrópicos: usar pitch=theta y dividir por 4*pi.
    No válido para electrones no relativistas ni como perfil de líneas.
    """
    sn=np.sin(np.deg2rad(theta_deg)); x=np.asarray(x)
    factor=np.sqrt(3)*E_CHARGE**3*sn/(4*np.pi*M_E*C*C)
    if distribution=='monoenergetica':
        g=1+kinetic_keV/MEC2
        return factor*synchrotron_F(x/(1.5*g*g*sn))
    nodes,weights=quadrature(nquad)
    theta=kT/MEC2
    lower=np.log(theta*1e-8); upper=np.log(theta*40)
    t=np.exp(lower+(nodes+1)/2*(upper-lower)); g=1+t
    weight=weights*(upper-lower)/2*t*gamma_pdf(g,kT)
    return factor*(synchrotron_F(x[:,None]/(1.5*g[None,:]**2*sn))@weight)

def broaden_log(y,dlog,sigma_ln_B=0.,sigma_ln_instrument=0.):
    """y=nu*dP/dnu/dOmega. Campo B0*exp(u), u~N(0,sigma_B²).
    El kernel físico incluye B²: <y_B>=exp(2*s²)*G_s[y_0](lognu-2*s²).
    El instrumental es una convolución de potencia en lognu.
    """
    y=np.asarray(y,float); result=y.copy(); index=np.arange(len(y))
    if sigma_ln_B>0:
        smooth=gaussian_filter1d(result,sigma_ln_B/dlog,mode='constant',truncate=6)
        result=np.exp(2*sigma_ln_B**2)*np.interp(
            index-2*sigma_ln_B**2/dlog,index,smooth,left=0,right=0)
    if sigma_ln_instrument>0:
        result=gaussian_filter1d(result,sigma_ln_instrument/dlog,mode='constant',truncate=6)
    return result

def compute_band(B,nu_min,nu_max,theta_deg,kT,distribution,kinetic_keV,
                 mode='armonicos',nmax=100,show_orders=8,sigma_ln_B=0.,
                 sigma_ln_instrument=.015,nquad=80,points=1800):
    """Calcula una banda y los órdenes visibles. Los armónicos no se renormalizan.
    El resultado armónico conserva su truncamiento explícito n<=nmax.
    """
    values=[B,nu_min,nu_max,theta_deg,kT,kinetic_keV,sigma_ln_B,sigma_ln_instrument]
    if not np.all(np.isfinite(values)): raise ValueError('Parámetros no finitos.')
    if min(B,nu_min,kT,kinetic_keV)<=0 or nu_max<=nu_min: raise ValueError('Escalas positivas y banda creciente.')
    if not 1<=theta_deg<=89: raise ValueError('theta debe estar entre 1 y 89 grados.')
    if min(sigma_ln_B,sigma_ln_instrument)<0 or max(sigma_ln_B,sigma_ln_instrument)>.5:
        raise ValueError('Anchuras logarítmicas entre 0 y 0.5.')
    if int(nmax)!=nmax or not 1<=nmax<=2000: raise ValueError('nmax entero entre 1 y 2000.')
    if int(show_orders)!=show_orders or not 0<=show_orders<=min(nmax,20):
        raise ValueError('Mostrar entre 0 y min(nmax,20) órdenes.')
    if not 16<=nquad<=500 or not 300<=points<=10000: raise ValueError('nquad: 16–500; points: 300–10000.')
    if mode not in ['armonicos','sincrotron']: raise ValueError('Modo desconocido.')
    gmax=electron_limits(kT,distribution,kinetic_keV)
    # Margen para convoluciones sin contaminar los extremos de la banda mostrada.
    padding=6*np.hypot(sigma_ln_B,sigma_ln_instrument)+2*sigma_ln_B**2+.05
    dlog=np.log(nu_max/nu_min)/(points-1)
    extra=int(np.ceil(padding/dlog))
    lognu=np.log(nu_min)+np.arange(-extra,points+extra)*dlog
    nu=np.exp(lognu); x=nu/gyrofrequency(B)
    total=np.zeros_like(nu); tail=np.zeros_like(nu); individual={}
    if mode=='armonicos':
        for order in range(1,int(nmax)+1):
            raw=B*nu*harmonic_spectrum(x,order,theta_deg,kT,distribution,kinetic_keV,nquad)
            total+=raw
            if order>max(0,nmax-10): tail+=raw
            if order<=show_orders:
                individual[order]=broaden_log(raw,dlog,sigma_ln_B,sigma_ln_instrument)
    else:
        total=B*nu*continuum_spectrum(x,theta_deg,kT,distribution,kinetic_keV,max(200,nquad))
    total=broaden_log(total,dlog,sigma_ln_B,sigma_ln_instrument)
    tail=broaden_log(tail,dlog,sigma_ln_B,sigma_ln_instrument)
    keep=slice(extra,extra+points)
    total=total[keep]; tail=tail[keep]; nu=nu[keep]
    individual={n:v[keep] for n,v in individual.items()}
    integral=trapezoid(total,np.log(nu))
    fraction=trapezoid(tail,np.log(nu))/integral if integral>0 else 0.
    return dict(nu=nu,total=total,individual=individual,tail_fraction=fraction,
                B=B,nu_B=gyrofrequency(B),gamma_max=gmax,integral=integral)
