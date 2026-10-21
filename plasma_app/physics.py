"""Backend científico para la aplicación, independiente de Streamlit.
Usa los motores verificados de los notebooks y añade semillas sincrotrón
isótropas y muestreo de importancia para resolver la cola Compton.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import trapezoid
from numpy.polynomial.legendre import leggauss
import magnetic_compton as rcs

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
import plasma_model as plasma
import ciclosincrotron_model as synch

MODES=('Bremsstrahlung','Comptonización','Sincrotrón')

@dataclass(frozen=True)
class Parameters:
    mode: str = 'Comptonización'
    compton_model: str = 'thermal'
    rcs_B_G: float = 3e11
    rcs_sigma_ln_B: float = .2
    rcs_tau: float = 3.
    rcs_kT_keV: float = 5.
    rcs_gamma_bulk: float = 10.
    rcs_beam_angle_deg: float = 0.
    beam_seed: str = 'blackbody'
    seed_kT_keV: float = .4
    kT_keV: float = 50.
    radius_km: float = 100.
    ne_cm3: float = 1e17
    distance_kpc: float = 5.
    seed_min_keV: float = .01
    seed_max_keV: float = 1e4
    plot_min_keV: float = .1
    plot_max_keV: float = 1e4
    photon_index: float = 2.
    beam_min_keV: float = .2
    beam_max_keV: float = 20.
    beam_luminosity: float = 1e35
    B_G: float = 1e11
    nrel_cm3: float = 1e12
    electron_distribution: str = 'thermal'
    gamma_min: float = 100.
    gamma_max: float = 1e5
    electron_index: float = 2.5
    synch_kT_keV: float = 20.
    synch_model: str = 'harmonics'
    theta_deg: float = 75.
    harmonic_max: int = 100
    harmonic_show: int = 8
    sigma_ln_B: float = 0.
    kinetic_keV: float = 50.
    synch_nquad: int = 80
    packets: int = 60000
    bins: int = 100
    random_seed: int = 2026
    auto_energy_range: bool = True
    seed_grid: int = 1800

    @property
    def compton(self): return 'Comptonización' in self.mode
    @property
    def source(self):
        return 'synch' if 'Sincrotrón' in self.mode else ('brems' if 'Bremsstrahlung' in self.mode else 'beam')
    @property
    def tau(self): return self.ne_cm3*plasma.SIGMA_T*self.radius_km*1e5

    def validate(self):
        if self.mode not in MODES: raise ValueError('Modelo desconocido.')
        for name,value in asdict(self).items():
            if isinstance(value,(int,float)) and not np.isfinite(value): raise ValueError(f'{name} no es finito.')
        if min(self.radius_km,self.distance_kpc,self.plot_min_keV,self.seed_min_keV)<=0:
            raise ValueError('Radio, distancia y energías mínimas deben ser positivos.')
        if self.ne_cm3<0 or self.nrel_cm3<0: raise ValueError('Las densidades no pueden ser negativas.')
        if self.plot_max_keV<=self.plot_min_keV or self.seed_max_keV<=self.seed_min_keV:
            raise ValueError('El máximo de cada banda debe superar el mínimo.')
        if ((self.compton and self.compton_model=='thermal') or self.source=='brems') and not 1<=self.kT_keV<=300:
            raise ValueError('kT térmica admitida: 1–300 keV.')
        if self.compton and self.compton_model=='thermal' and self.tau>3:
            raise ValueError(f'La profundidad óptica radial es {self.tau:.2f}; debe ser ≤ 3. Reduzca ne o radio.')
        if self.compton_model not in ('thermal','magnetic'): raise ValueError('Modelo Compton desconocido.')
        if self.compton and self.compton_model=='magnetic':
            rcs.validate(self.rcs_B_G,self.rcs_sigma_ln_B,self.rcs_tau,self.rcs_kT_keV,self.rcs_gamma_bulk)
            if not 0<=self.rcs_beam_angle_deg<=180: raise ValueError('Ángulo haz–B: 0–180 grados.')
        if self.source=='beam':
            if self.beam_seed not in ('powerlaw','blackbody'): raise ValueError('Semilla desconocida.')
            if not .01<=self.seed_kT_keV<=10: raise ValueError('kT semilla: 0.01–10 keV.')
            if self.beam_min_keV<=0 or self.beam_max_keV<=self.beam_min_keV or self.beam_luminosity<=0:
                raise ValueError('Compruebe los límites y luminosidad del haz.')
            if not -2<=self.photon_index<=5: raise ValueError('Índice de fotones admitido: −2 a 5.')
        if self.source=='synch':
            if self.B_G<=0: raise ValueError('B debe ser positivo.')
            if self.synch_model not in ('harmonics','continuum'): raise ValueError('Modelo sincrotrón desconocido.')
            if self.electron_distribution not in ('mono','thermal'): raise ValueError('Elija electrones térmicos o monoenergéticos.')
            if not 1<=self.synch_kT_keV<=1e6: raise ValueError('kT: 1–10⁶ keV.')
            if not 0<self.kinetic_keV<=1e9: raise ValueError('Energía cinética: 0–10⁹ keV.')
            if not 1<=self.theta_deg<=89: raise ValueError('Ángulo: 1–89 grados.')
            if not 10<=self.harmonic_max<=2000 or int(self.harmonic_max)!=self.harmonic_max: raise ValueError('nmax entero: 10–2000.')
            if not 0<=self.harmonic_show<=min(20,self.harmonic_max): raise ValueError('Órdenes visibles: 0–20 y ≤ nmax.')
            if not 0<=self.sigma_ln_B<=.5: raise ValueError('σ ln B: 0–0.5.')
            if not 16<=self.synch_nquad<=500: raise ValueError('Cuadratura: 16–500 nodos.')
        if int(self.packets)!=self.packets or not 2000<=self.packets<=1000000: raise ValueError('Paquetes: entero 2000–10⁶.')
        if int(self.bins)!=self.bins or not 30<=self.bins<=300: raise ValueError('Bins: entero 30–300.')
        if int(self.seed_grid)!=self.seed_grid or not 100<=self.seed_grid<=4000: raise ValueError('Malla semilla: 100–4000 puntos.')
        if int(self.random_seed)!=self.random_seed or not 0<=self.random_seed<2**32: raise ValueError('Semilla aleatoria: entero 0–2³²−1.')

def electron_quadrature(p,order=120):
    if p.electron_distribution=='mono': return np.array([p.gamma_min]),np.array([1.])
    nodes,weights=leggauss(order)
    if p.electron_distribution=='powerlaw':
        low,high=np.log(p.gamma_min),np.log(p.gamma_max)
        g=np.exp(low+(nodes+1)/2*(high-low))
        weight=weights*(high-low)/2*g*g**(-p.electron_index)
    else:
        theta=p.synch_kT_keV/plasma.MEC2
        low,high=np.log(theta*1e-8),np.log(theta*40)
        t=np.exp(low+(nodes+1)/2*(high-low)); g=1+t
        weight=weights*(high-low)/2*t*synch.gamma_pdf(g,p.synch_kT_keV)
    return g,weight/weight.sum()

def synch_photon_luminosity(E,p,order=120,pitch_order=32):
    """Q_E [fotones/s/keV]. Campo enmarañado: promedio isotrópico de pitch.
    P_nu= sqrt(3)e³ B sin(alpha)/(m_e c²) F(nu/nu_c), integrado en ángulo.
    """
    gamma,weight=electron_quadrature(p,order)
    mu,pitch_weights=leggauss(pitch_order)
    pitch_weights/=2
    nu=np.asarray(E)*plasma.KEV/synch.H
    pnu=np.zeros(len(nu))
    factor=np.sqrt(3)*synch.E_CHARGE**3*p.B_G/(synch.M_E*synch.C**2)
    for m,w in zip(mu,pitch_weights):
        sn=np.sqrt(1-m*m)
        critical=1.5*synch.gyrofrequency(p.B_G)*gamma**2*sn
        pnu+=w*factor*sn*(synch.synchrotron_F(nu[:,None]/critical[None,:])@weight)
    volume=4*np.pi/3*(p.radius_km*1e5)**3
    # L_E = L_nu * dnu/dE; Q_E = L_E/(E*keV_erg).
    return pnu*(plasma.KEV/synch.H)*p.nrel_cm3*volume/(np.asarray(E)*plasma.KEV)

def interpolate_photons(E,grid,Q):
    """Interpolación log-log positiva; cero fuera de la banda semilla."""
    E=np.asarray(E)
    idx=np.clip(np.searchsorted(grid,E,side='right')-1,0,len(grid)-2)
    fraction=(np.log(E)-np.log(grid[idx]))/(np.log(grid[idx+1])-np.log(grid[idx]))
    left=Q[idx]; right=Q[idx+1]
    both=(left>0)&(right>0)
    result=(1-fraction)*left+fraction*right
    result=np.where(both,np.exp((1-fraction)*np.log(np.maximum(left,1e-300))+
                               fraction*np.log(np.maximum(right,1e-300))),result)
    return np.where((E>=grid[0])&(E<=grid[-1]),np.maximum(result,0),0)


def harmonic_bundle(p,low,high,points):
    """Armónicos del notebook. Q equivalente isotrópica para una dirección;
    en modo combinado se integra además sobre la dirección del observador.
    """
    if p.compton:
        mu,weights=leggauss(8)
        angles=np.degrees(np.arccos((mu+1)/2))
        weights=weights/2
    else:
        angles=[p.theta_deg]; weights=[1.]
    volume=4*np.pi/3*(p.radius_km*1e5)**3
    total=None; individual={}; tail=0.
    for angle,weight in zip(angles,weights):
        band=synch.compute_band(p.B_G,low*plasma.KEV/synch.H,high*plasma.KEV/synch.H,
             float(angle),p.synch_kT_keV,'termica' if p.electron_distribution=='thermal' else 'monoenergetica',
             p.kinetic_keV,mode='armonicos' if p.synch_model=='harmonics' else 'sincrotron',
             nmax=p.harmonic_max,show_orders=p.harmonic_show,sigma_ln_B=p.sigma_ln_B,
             sigma_ln_instrument=0.,points=max(300,points),nquad=p.synch_nquad)
        energy=band['nu']*synch.H/plasma.KEV
        # nu Pnu -> photon luminosity per keV.
        factor=4*np.pi*p.nrel_cm3*volume/(energy**2*plasma.KEV)
        q=band['total']*factor
        total=weight*q if total is None else total+weight*q
        for n,y in band['individual'].items():
            individual[n]=individual.get(n,0)+weight*y*factor
        tail+=weight*band['tail_fraction']
    return energy,total,individual,tail

def seed_spectrum(p):
    if p.source=='beam':
        grid=np.geomspace(p.beam_min_keV,p.beam_max_keV,p.seed_grid)
        if p.beam_seed=='blackbody':
            x=grid/p.seed_kT_keV
            shape=grid**2*np.exp(-x)/(-np.expm1(-x))
        else:
            shape=(grid/np.sqrt(grid[0]*grid[-1]))**(-p.photon_index)
        if not np.any(shape>0): raise ValueError('La banda semilla no contiene potencia de cuerpo negro representable.')
        Q=shape*p.beam_luminosity/(plasma.KEV*trapezoid(grid*shape,grid))
    elif p.source=='synch':
        grid,Q,_,_=harmonic_bundle(p,p.seed_min_keV,p.seed_max_keV,p.seed_grid)
    else:
        grid=np.geomspace(p.seed_min_keV,p.seed_max_keV,p.seed_grid)
        if p.source=='brems': Q=plasma.brems(grid,p.kT_keV,p.ne_cm3,p.radius_km)
        else: Q=synch_photon_luminosity(grid,p)
    return grid,Q

def importance_packets(grid,Q,N,rng):
    """q(E)=1/[E ln(Emax/Emin)]. Peso por paquete = Q(E)/(N q(E))."""
    width=np.log(grid[-1]/grid[0])
    E=np.exp(rng.uniform(np.log(grid[0]),np.log(grid[-1]),N))
    rate=interpolate_photons(E,grid,Q)*E*width/N
    return E,rate

def weighted_histogram(E,rate,bins,distance_kpc,mask=None):
    """Flujo F_E y error estándar de importance sampling (N fijo)."""
    N=len(E)
    contribution=rate*E*plasma.KEV/(4*np.pi*(distance_kpc*plasma.KPC)**2)
    if mask is not None: contribution=np.where(mask,contribution,0)
    idx=np.searchsorted(bins,E,side='right')-1
    idx=np.where(E==bins[-1],len(bins)-2,idx)
    valid=(idx>=0)&(idx<len(bins)-1)
    total=np.bincount(idx[valid],weights=contribution[valid],minlength=len(bins)-1)
    square=np.bincount(idx[valid],weights=contribution[valid]**2,minlength=len(bins)-1)
    # rate ya incluye 1/N. Var(media)= (sum w_i² - total²/N)*N/(N-1).
    variance=np.maximum(0,(square-total*total/N)*N/(N-1))
    width=np.diff(bins)
    return total/width,np.sqrt(variance)/width

def calculate(p):
    p.validate()
    harmonic_data=None
    if p.source=='synch':
        harmonic_data=harmonic_bundle(p,p.seed_min_keV,p.seed_max_keV,p.seed_grid)
        grid,Q,orders,tail=harmonic_data
    else:
        grid,Q=seed_spectrum(p)
    bins=np.geomspace(p.plot_min_keV,p.plot_max_keV,p.bins+1)
    E=np.sqrt(bins[:-1]*bins[1:])
    factor=4*np.pi*(p.distance_kpc*plasma.KPC)**2
    intrinsic=interpolate_photons(E,grid,Q)*E*plasma.KEV/factor
    zero=np.zeros_like(E)
    out=dict(energy=E,bins=bins,intrinsic=intrinsic,emergent=intrinsic.copy(),direct=intrinsic.copy(),
             scattered=zero.copy(),upscattered=zero.copy(),error=zero.copy(),seed_mc=zero.copy(),
             parameters=asdict(p),messages=[],tau=p.tau)
    if harmonic_data is not None:
        out['harmonic_energy']=grid
        out['harmonic_total']=Q*grid*plasma.KEV/factor
        out['harmonics']={n:y*grid*plasma.KEV/factor for n,y in orders.items()}
        out['harmonic_tail']=tail
        if tail>.01: out['messages'].append('Los últimos diez órdenes aportan más del 1 %: aumente nmax para comprobar convergencia.')
        eb=synch.H*synch.gyrofrequency(p.B_G)/plasma.KEV
        escale=p.synch_kT_keV if p.electron_distribution=='thermal' else p.kinetic_keV
        if eb/escale>.1: out['messages'].append('La cuantización de Landau puede importar (EB/kT > 0.1); modelo clásico ilustrativo.')
        gmax=1+(40*p.synch_kT_keV if p.electron_distribution=='thermal' else p.kinetic_keV)/plasma.MEC2
        if gmax*p.B_G/synch.B_Q>.05:
            out['messages'].append('La cola electrónica puede requerir correcciones cuánticas; no están incluidas.')
    out['seed_luminosity']=float(trapezoid(grid*Q,grid)*plasma.KEV)
    if p.source!='beam':
        out['messages'].append('La emisión semilla se integra solo en la banda indicada. Amplíela para comprobar convergencia.')
        edge_fraction=(trapezoid(grid[:30]*Q[:30],grid[:30])+trapezoid(grid[-30:]*Q[-30:],grid[-30:]))/max(out['seed_luminosity']/plasma.KEV,1e-300)
        if edge_fraction>.05:
            out['messages'].append('Hay potencia relevante cerca de los extremos de la malla semilla; revise sus límites.')
    if p.source=='synch':
        out['messages'].append(('Armónicos clásicos' if p.synch_model=='harmonics' else 'Continuo sincrotrón ultrarrelativista') + ': mismo motor del notebook. ' +
            ('La semilla Compton se promedia sobre las direcciones de emisión (8 nodos angulares); las curvas muestran ese promedio.' if p.compton else
             'Se muestra la dirección seleccionada; la luminosidad indicada es equivalente isotrópica en esa dirección.'))
    if p.source=='synch' and p.synch_model=='continuum':
        relativistic=(p.synch_kT_keV/plasma.MEC2>=3 if p.electron_distribution=='thermal' else 1+p.kinetic_keV/plasma.MEC2>=10)
        if not relativistic: out['messages'].append('Continuo fuera del régimen ultrarrelativista: use armónicos o aumente la energía electrónica.')
    if not p.compton:
        out.update(energy_ratio=1.,scattered_fraction=0.,upscattered_fraction=0.,
                   energy_exchange=0.,outside_energy_fraction=0.,packet_number_ratio=1.)
        return out
    if not np.any(Q>0):
        out.update(energy_ratio=1.,scattered_fraction=0.,upscattered_fraction=0.,
                   energy_exchange=0.,outside_energy_fraction=0.,packet_number_ratio=1.)
        out['messages'].append('La luminosidad semilla es cero con los parámetros elegidos.')
        return out
    rng=np.random.default_rng(p.random_seed)
    initial,rate=importance_packets(grid,Q,p.packets,rng)
    if p.compton_model=='thermal':
        final,counts=plasma.transport(initial,p.kT_keV,p.tau,rng,
                                     source='beam' if p.source=='beam' else 'internal')
    else:
        final,counts,stats=rcs.transport(initial,p.rcs_B_G,p.rcs_sigma_ln_B,p.rcs_tau,
            p.rcs_kT_keV,p.rcs_gamma_bulk,rng,source='beam' if p.source=='beam' else 'internal',
            beam_angle_deg=p.rcs_beam_angle_deg)
        out['rcs_stats']=stats
        out['rcs_density_cm3']=p.rcs_tau/(rcs.peak_cross_section(p.rcs_B_G,p.rcs_sigma_ln_B)*p.radius_km*1e5)
        out['rcs_EB_keV']=plasma.MEC2*p.rcs_B_G/rcs.BQ
        out['messages'].append('RCS local en límite Thomson: resonancia promediada sobre una mezcla lognormal de B, dos flujos opuestos a lo largo del campo y temperatura paralela. La profundidad de referencia resonante NO es τ Thomson.')
        out['messages'].append('Demostrador magnetizado, no modelo QED de magnetosfera global: sin recoil, polarización persistente, dipolo/twist, pares, gravedad ni equilibrio del flujo. No usar para inferir parámetros de magnetares a partir de una cola dura sin ampliar el modelo.')
    if p.auto_energy_range:
        meaningful=rate>0
        lo=min(p.plot_min_keV,float(initial[meaningful].min())*.9,float(final[meaningful].min())*.9)
        hi=max(p.plot_max_keV,float(initial[meaningful].max())*1.1,float(final[meaningful].max())*1.1)
        bins=np.geomspace(lo,hi,p.bins+1); E=np.sqrt(bins[:-1]*bins[1:])
        out['energy']=E; out['bins']=bins
        out['intrinsic']=interpolate_photons(E,grid,Q)*E*plasma.KEV/factor
    def photon_hist(energies,mask=None):
        indices=np.searchsorted(bins,energies,side='right')-1
        valid=(indices>=0)&(indices<len(bins)-1)
        if mask is not None: valid &= mask
        return np.bincount(indices[valid],weights=rate[valid],minlength=len(bins)-1)/(factor*np.diff(bins))
    out['seed_photons']=photon_hist(initial)
    out['emergent_photons']=photon_hist(final)
    out['upscattered_photons']=photon_hist(final,(counts>0)&(final>initial))
    out['photon_excess']=out['emergent_photons']-out['seed_photons']
    out['max_escaped_keV']=float(final[rate>0].max())
    out['beyond_seed_fraction']=float(rate[final>grid[-1]].sum()/rate.sum())
    out['seed_mc'],_=weighted_histogram(initial,rate,bins,p.distance_kpc)
    out['emergent'],out['error']=weighted_histogram(final,rate,bins,p.distance_kpc)
    out['direct'],_=weighted_histogram(final,rate,bins,p.distance_kpc,counts==0)
    out['scattered'],_=weighted_histogram(final,rate,bins,p.distance_kpc,counts>0)
    out['upscattered'],_=weighted_histogram(final,rate,bins,p.distance_kpc,(counts>0)&(final>initial))
    input_power=float(np.sum(rate*initial)*plasma.KEV)
    output_power=float(np.sum(rate*final)*plasma.KEV)
    outside=(final<bins[0])|(final>bins[-1])
    out.update(energy_ratio=output_power/input_power,
        energy_exchange=output_power-input_power,
        scattered_fraction=float(np.sum(rate[counts>0])/rate.sum()),
        upscattered_fraction=float(np.sum(rate[(counts>0)&(final>initial)])/rate.sum()),
        outside_energy_fraction=float(np.sum(rate[outside]*final[outside])/np.sum(rate*final)),
        packet_number_ratio=1.,seed_mc_luminosity=input_power,
        emergent_mc_luminosity=output_power,
        hard_luminosity=float(np.sum(rate[final>=100]*final[final>=100])*plasma.KEV),
        seed_hard_luminosity=float(np.sum(rate[initial>=100]*initial[initial>=100])*plasma.KEV))
    positive=out['emergent']>0
    if np.any(positive & (out['error']>.3*out['emergent'])):
        out['messages'].append('Algunos bins tienen error Monte Carlo >30 %. Aumente los paquetes para resolver mejor la cola.')
    if out['outside_energy_fraction']>.01:
        out['messages'].append(f'{out["outside_energy_fraction"]:.1%} de la potencia emergente está fuera de la banda visible. Amplíela; esos fotones sí se transportan.')
    out['messages'].append('Compton conserva el número de fotones; los electrones a distribución impuesta pueden aportar o absorber energía.')
    if bins[-1]>511 and p.compton_model=='thermal':
        out['messages'].append('Por encima de 511 keV se mantiene Klein–Nishina, pero no se incluyen producción de pares ni cascadas.')
    return out
