"""Local resonant cyclotron scattering (RCS), conservative Thomson limit.

Not a global magnetosphere or a magnetic QED Compton solver. Electrons occupy
an effective 1-D distribution along a fixed field axis. Unpolarized closure:
polarization memory is discarded at each scattering. B magnitudes are a
micro-mixed truncated lognormal distribution, independent of position.

sigma_res(E',mu') = C (1+mu'^2) p_EB(E'), C=pi² r_e c hbar / keV_erg.
This is the delta resonance averaged over B, not an arbitrary KN multiplier.
See Nobili et al. 2008 doi:10.1111/j.1365-2966.2008.13125.x;
Yamasaki et al. 2020 doi:10.1093/mnras/staa2223 (cross-section only).
"""
from functools import lru_cache
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf

MEC2=510.99895
BQ=4.414005e13
C_LIGHT=2.99792458e10
RE=2.8179403262e-13
HBAR=1.054571817e-27
KEV=1.602176634e-9
RESONANCE_AREA=np.pi**2*RE*C_LIGHT*HBAR/KEV # cm² keV
TRUNCATION=4.


def field_pdf(E,B,sigma):
    """p(E_B) per keV, truncated at +/-4 sigma in ln B."""
    E=np.asarray(E,float); center=MEC2*B/BQ
    z=np.log(np.maximum(E,1e-300)/center)/sigma
    norm=erf(TRUNCATION/np.sqrt(2))
    return np.where(np.abs(z)<=TRUNCATION,
                    np.exp(-.5*z*z)/(np.maximum(E,1e-300)*sigma*np.sqrt(2*np.pi)*norm),0.)


def peak_cross_section(B,sigma):
    center=MEC2*B/BQ
    mode=center*np.exp(-sigma*sigma)
    return float(2*RESONANCE_AREA*field_pdf(mode,B,sigma))


def resonance_ratio(Erest,mu_rest,B,sigma):
    """sigma/sigma_peak; lies in [0,1]."""
    return RESONANCE_AREA*(1+mu_rest**2)*field_pdf(Erest,B,sigma)/peak_cross_section(B,sigma)


@lru_cache(maxsize=32)
def parallel_distribution(kT):
    """Normalized CDF in rest-frame parallel momentum p/(m_e c).
    f(p) proportional to exp[-(sqrt(1+p²)-1)/theta].
    """
    theta=kT/MEC2
    pmax=np.sqrt((1+40*theta)**2-1)
    momentum=np.linspace(-pmax,pmax,16385)
    pdf=np.exp(-(np.sqrt(1+momentum**2)-1)/theta)
    cdf=cumulative_trapezoid(pdf,momentum,initial=0); cdf/=cdf[-1]
    return momentum,cdf


def sample_parallel_beta(n,kT,gamma_bulk,rng):
    """Equal counterstreams in the lab. kT in each stream rest frame.
    Jacobian dp_lab/dp_rest = gamma_bulk*(1+beta_bulk*beta_rest).
    """
    beta_bulk=np.sqrt(1-1/gamma_bulk**2)
    p,cdf=parallel_distribution(kT)
    result=np.empty(n); pending=np.arange(n)
    while len(pending):
        pp=np.interp(rng.random(len(pending)),cdf,p)
        gg=np.sqrt(1+pp*pp); bb=pp/gg
        accept=rng.random(len(pending)) < (1+beta_bulk*bb)/(1+beta_bulk)
        ids=pending[accept]
        result[ids]=(bb[accept]+beta_bulk)/(1+bb[accept]*beta_bulk)
        pending=pending[~accept]
    return result*rng.choice(np.array([-1.,1.]),size=n)


def sample_mu(n,rng):
    """Unpolarized outgoing mu relative to B: p(mu)=3(1+mu²)/8."""
    mu=np.empty(n); todo=np.arange(n)
    while len(todo):
        trial=rng.uniform(-1,1,len(todo)); accepted=rng.random(len(todo))<(1+trial*trial)/2
        mu[todo[accepted]]=trial[accepted]; todo=todo[~accepted]
    return mu


def scatter(E,direction,beta,rng):
    """Coherent in the electron frame; exact Doppler transformations."""
    gamma=1/np.sqrt(1-beta*beta)
    Erest=gamma*E*(1-beta*direction[:,2])
    mu=sample_mu(len(E),rng); phi=rng.uniform(0,2*np.pi,len(E))
    denominator=1+beta*mu
    final_E=gamma*Erest*denominator
    perp=np.sqrt(1-mu*mu)/(gamma*denominator)
    final_direction=np.column_stack((perp*np.cos(phi),perp*np.sin(phi),(mu+beta)/denominator))
    return final_E,final_direction,Erest


def validate(B,sigma,tau,kT,gamma_bulk):
    if not np.all(np.isfinite([B,sigma,tau,kT,gamma_bulk])): raise ValueError('Parámetros RCS no finitos.')
    if not .03<=sigma<=.5: raise ValueError('Anchura local σ ln B: 0.03–0.5.')
    if B<=0 or B*np.exp(TRUNCATION*sigma)/BQ>.03:
        raise ValueError('RCS Thomson requiere B_local × exp(4 σ ln B) ≤ 0.03 BQ = 1.32×10¹² G. Reduzca B o su dispersión.')
    if not 0<=tau<=10: raise ValueError('τ resonante de referencia: 0–10.')
    if not .01<=kT<=100: raise ValueError('kT paralelo RCS: 0.01–100 keV.')
    if not 1<=gamma_bulk<=30: raise ValueError('Gamma colectivo RCS: 1–30.')


def transport(E,B,sigma,tau,kT,gamma_bulk,rng,source='beam',beam_angle_deg=0.):
    """Sphere, radius=1. tau=n_charge sigma_peak R; not Thomson depth.
    Null-collision bound 2*tau includes the relative-flux factor 1-beta*mu.
    All packet weights and photon numbers are unchanged.
    """
    validate(B,sigma,tau,kT,gamma_bulk)
    if source not in ('beam','internal'): raise ValueError('Geometría RCS desconocida.')
    if not 0<=beam_angle_deg<=180: raise ValueError('Ángulo haz–B: 0–180 grados.')
    E=np.asarray(E,float).copy()
    if np.any(~np.isfinite(E)) or np.any(E<=0): raise ValueError('Fotones con energías positivas y finitas requeridos.')
    N=len(E); counts=np.zeros(N,int)
    stats={'max_rest_energy_keV':0.,'collision_count':0}
    if tau==0: return E,counts,stats
    if source=='beam':
        angle=np.deg2rad(beam_angle_deg)
        beam=np.array([np.sin(angle),0.,np.cos(angle)])
        u=np.array([np.cos(angle),0.,-np.sin(angle)]); v=np.array([0.,1.,0.])
        radius=np.sqrt(rng.random(N)); phi=rng.uniform(0,2*np.pi,N)
        pos=radius[:,None]*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v)-np.sqrt(1-radius*radius)[:,None]*beam
        direction=np.tile(beam,(N,1))
    else:
        pos=rng.normal(size=(N,3)); pos/=np.linalg.norm(pos,axis=1)[:,None]
        pos*=rng.random(N)[:,None]**(1/3)
        direction=rng.normal(size=(N,3)); direction/=np.linalg.norm(direction,axis=1)[:,None]
    active=np.arange(N)
    for _ in range(20000):
        if not len(active): return E,counts,stats
        dot=np.sum(pos[active]*direction[active],axis=1)
        distance=-dot+np.sqrt(np.maximum(0,dot*dot+1-np.sum(pos[active]**2,axis=1)))
        step=rng.exponential(1/(2*tau),len(active))
        inside=step<distance; active=active[inside]; step=step[inside]
        pos[active]+=step[:,None]*direction[active]
        beta=sample_parallel_beta(len(active),kT,gamma_bulk,rng)
        mu=direction[active,2]; relative=1-beta*mu; gamma=1/np.sqrt(1-beta*beta)
        er=gamma*E[active]*relative
        mur=(mu-beta)/relative
        probability=relative*resonance_ratio(er,mur,B,sigma)/2
        if np.any(probability>1+1e-12): raise ArithmeticError('Cota de colisión RCS inválida.')
        accepted=rng.random(len(active))<probability
        ids=active[accepted]
        if len(ids):
            E[ids],direction[ids],rest=scatter(E[ids],direction[ids],beta[accepted],rng)
            counts[ids]+=1
            stats['collision_count']+=len(ids)
            stats['max_rest_energy_keV']=max(stats['max_rest_energy_keV'],float(rest.max()))
    raise RuntimeError('Transporte RCS no convergió; no se devuelven fotones truncados.')
