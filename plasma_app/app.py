"""streamlit run plasma_app/app.py"""
from dataclasses import asdict
from io import BytesIO, StringIO
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from physics import Parameters, MODES, calculate

st.set_page_config(page_title='Laboratorio de espectros de altas energías',page_icon='✦',layout='wide')
st.markdown('''<style>
.block-container {padding-top:2rem; max-width:1450px}
[data-testid="stMetric"] {background:#f4f7fb; padding:14px 18px; border-radius:12px; color:#14253b}
[data-testid="stMetricLabel"] {color:#45586c}
h1 {letter-spacing:-.045em}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False,max_entries=12)
def run_model(values,version=6):
    return calculate(Parameters(**values))

with st.sidebar:
    st.title('✦ Modelos radiativos')
    if st.session_state.get('mode', 'Comptonización') not in MODES:
        st.session_state['mode']='Comptonización'
    mode=st.selectbox('Proceso',MODES,index=1,key='mode')
    source='synch' if 'Sincrotrón' in mode else ('brems' if 'Bremsstrahlung' in mode else 'beam')
    compton='Comptonización' in mode
    compton_model='thermal'; beam_seed='blackbody'
    if compton:
        compton_model=st.selectbox('Modelo de comptonización',['thermal','magnetic'],
            format_func=lambda x:'Térmico · Klein–Nishina' if x=='thermal' else 'Magnetizado · RCS Thomson local',key='compton_model')
        if compton_model=='magnetic':
            st.info('Aproximación resonante local subcrítica; no es un modelo QED completo de magnetar.')
    if source=='beam':
        beam_seed=st.selectbox('Espectro semilla',['powerlaw','blackbody'],
            index=1,
            format_func=lambda x:'Ley de potencias' if x=='powerlaw' else 'Cuerpo negro',key='seed_'+compton_model)
    distribution='thermal'; synch_model='harmonics'
    if source=='synch':
        synch_model=st.selectbox('Modelo ciclotrón–sincrotrón',['harmonics','continuum'],
            format_func=lambda x:'Armónicos clásicos' if x=='harmonics' else 'Continuo sincrotrón (ultrarrelativista)',key='synch_model')
        distribution=st.selectbox('Distribución electrónica',['thermal','mono'],
            format_func=lambda x:'Maxwell–Jüttner' if x=='thermal' else 'Monoenergética',key='synch_distribution')
    with st.form('parameters'):
        st.caption('GEOMETRÍA Y NORMALIZACIÓN')
        radius=st.number_input('Radio de la esfera [km]',min_value=.01,max_value=1e9,value=100.,format='%.3g')
        distance=st.number_input('Distancia [kpc]',min_value=.000001,max_value=1e6,value=5.,format='%.4g')
        ne=1e17; kt=50.
        if (compton and compton_model=='thermal') or source=='brems':
            ne=st.number_input('Densidad térmica ne [cm⁻³]',min_value=0.,max_value=1e25,value=1e17,format='%.4e')
            kt=st.number_input('kT plasma térmico [keV]',min_value=1.,max_value=300.,value=50.,step=5.)
        rcsB=3e11; rcswidth=.2; rcstau=3.; rcskt=5.; rcsgamma=10.; rcsangle=0.; seedkt=.4
        if compton and compton_model=='magnetic':
            st.caption('CAPA RESONANTE LOCAL · B NO ES EL CAMPO SUPERFICIAL')
            rcsB=st.number_input('Campo local RCS [G]',min_value=1e7,max_value=1.3e12,value=3e11,format='%.4g')
            rcswidth=st.slider('Dispersión local RCS σ ln B',.03,.5,.2,.01)
            rcstau=st.slider('τ resonante de referencia',0.,10.,3.,.1,help='n_cargas × sección eficaz máxima × radio. No es la profundidad Thomson ni la τ real de cada fotón.')
            rcskt=st.number_input('kT paralelo RCS [keV]',min_value=.01,max_value=100.,value=5.,step=1.)
            rcsgamma=st.number_input('Gamma colectivo (dos contraflujos)',min_value=1.,max_value=30.,value=10.,step=1.)
            if source=='beam':
                rcsangle=st.slider('Ángulo haz–B local [°]',0.,180.,0.,5.)
        beam_min=.2; beam_max=20.; index=2.; luminosity=1e35
        B=1e11; nrel=1e12; gmin=100.; gmax=1e5; eindex=2.5; synkt=20.
        theta=75.; nmax=100; nshow=8; sigmab=0.; kinetic=50.; nquad=80; gridpoints=1800
        seedmin=.01; seedmax=1e4
        if source=='beam':
            st.caption('HAZ INCIDENTE · '+('dN/dE ∝ E⁻Γ' if beam_seed=='powerlaw' else 'CUERPO NEGRO'))
            beam_min=st.number_input('E mínima del haz [keV]',min_value=1e-6,max_value=1e6,value=.2,format='%.4g')
            beam_max=st.number_input('E máxima del haz [keV]',min_value=1e-5,max_value=1e8,value=20.,format='%.4g')
            if beam_seed=='powerlaw':
                index=st.number_input('Índice de fotones Γ',min_value=-2.,max_value=5.,value=2.,step=.1)
            else:
                seedkt=st.number_input('kT cuerpo negro semilla [keV]',min_value=.01,max_value=10.,value=.4,step=.05)
            luminosity=st.number_input('Luminosidad interceptada [erg/s]',min_value=1.,max_value=1e50,value=1e35,format='%.4e')
        if source=='synch':
            st.caption('CICLOTRÓN–SINCROTRÓN · MODELO DEL NOTEBOOK')
            B=st.number_input('Campo magnético B [G]',min_value=1e-9,max_value=1e12,value=1e11 if synch_model=='harmonics' else 1e9,format='%.4g',key='B_'+synch_model)
            nrel=st.number_input('Densidad de electrones emisores [cm⁻³]',min_value=0.,max_value=1e25,value=1e12,format='%.4e')
            if distribution=='thermal':
                synkt=st.number_input('kT electrones emisores [keV]',min_value=1.,max_value=1e6,value=20. if synch_model=='harmonics' else 5000.,step=10.,key='kT_'+synch_model)
            else:
                kinetic=st.number_input('Energía cinética del electrón [keV]',min_value=.01,max_value=1e9,value=50. if synch_model=='harmonics' else 10000.,format='%.4g',key='kinetic_'+synch_model)
            if not compton:
                theta=st.slider('Ángulo observador–B [°]',1.,89.,75.,1.)
            else:
                st.caption('Semilla promediada sobre direcciones; kT de la corona se controla arriba.')
            if synch_model=='harmonics':
                nmax=st.select_slider('Orden máximo nmax',options=[30,50,100,200,300,500,1000,2000],value=100)
                nshow=st.slider('Órdenes individuales visibles',0,20,8)
            else:
                nshow=0
                st.caption('Continuo ultrarrelativista: sustituye la suma de armónicos, no se suma a ella.')
            sigmab=st.slider('Dispersión del campo σ ln B',0.,.5,0.,.01)
            nquad=st.number_input('Nodos de cuadratura electrónica',min_value=16,max_value=500,value=80,step=16)
            gridpoints=st.number_input('Puntos de la malla semilla',min_value=300,max_value=4000,value=1800,step=100)
            st.caption('Pruebe kT = 20, 50 y 100 keV manteniendo B y ángulo: los armónicos se ensanchan y solapan.')
        if source!='beam':
            st.caption('BANDA DE FOTONES SEMILLA')
            seedmin=st.number_input('Semilla mínima [keV]',min_value=1e-8,max_value=1e5,value=.01,format='%.4g')
            seedmax=st.number_input('Semilla máxima [keV]',min_value=1e-5,max_value=1e8,value=1e4,format='%.4g')
        st.caption('BANDA VISIBLE · RAYOS X Y GAMMA')
        emin=st.number_input('Mostrar desde [keV]',min_value=1e-6,max_value=1e6,value=.1,format='%.4g')
        emax=st.number_input('Mostrar hasta [keV]',min_value=1e-5,max_value=1e8,value=100. if source=='synch' and not compton else 1e4,format='%.4g',key='plot_max_'+mode)
        bins=st.slider('Número de bins',30,300,100,10)
        packets=60000; seed=2026; auto_range=True
        if compton:
            auto_range=st.checkbox('Ampliar la banda para incluir todos los fotones simulados',value=True)
            packets=st.select_slider('Paquetes Monte Carlo',options=[2000,10000,30000,60000,150000,300000,1000000],value=60000)
            seed=st.number_input('Semilla aleatoria',min_value=0,max_value=2**32-1,value=2026,step=1)
        submitted=st.form_submit_button('Calcular espectro',type='primary',width='stretch')
    st.caption('Los parámetros del formulario se aplican al pulsar Calcular. El cambio de vista no repite la simulación.')

p=Parameters(mode=mode,compton_model=compton_model,rcs_B_G=rcsB,rcs_sigma_ln_B=rcswidth,
    rcs_tau=rcstau,rcs_kT_keV=rcskt,rcs_gamma_bulk=rcsgamma,rcs_beam_angle_deg=rcsangle,
    beam_seed=beam_seed,seed_kT_keV=seedkt,kT_keV=kt,radius_km=radius,ne_cm3=ne,distance_kpc=distance,
    seed_min_keV=seedmin,seed_max_keV=seedmax,plot_min_keV=emin,plot_max_keV=emax,
    photon_index=index,beam_min_keV=beam_min,beam_max_keV=beam_max,beam_luminosity=luminosity,
    B_G=B,nrel_cm3=nrel,electron_distribution=distribution,gamma_min=gmin,gamma_max=gmax,
    electron_index=eindex,synch_kT_keV=synkt,theta_deg=theta,harmonic_max=nmax,
    harmonic_show=nshow,sigma_ln_B=sigmab,synch_model=synch_model,kinetic_keV=kinetic,
    synch_nquad=nquad,seed_grid=gridpoints,packets=packets,bins=bins,random_seed=seed,auto_energy_range=auto_range)

st.caption('PLASMA LAB / EMISIÓN Y TRANSFERENCIA')
st.title('Laboratorio de espectros de altas energías')
st.write('Explora la emisión con tres de los modelos vistos en clase. ')

if 'result' in st.session_state and (
    'compton_model' not in st.session_state.result['parameters']
    or st.session_state.result['parameters']['mode'] not in MODES
):
    del st.session_state.result
if submitted or 'result' not in st.session_state:
    try:
        with st.spinner('Calculando emisión y transporte de fotones…'):
            st.session_state.result=run_model(asdict(p))
        st.session_state.pop('calculation_error',None)
    except (ValueError,ArithmeticError,RuntimeError) as exc:
        st.session_state.calculation_error=str(exc)
if 'calculation_error' in st.session_state:
    st.error(st.session_state.calculation_error)
if 'result' not in st.session_state:
    st.info('Revise los parámetros y pulse Calcular espectro.')
    st.stop()
r=st.session_state.result
if r['parameters']['compton_model']!=compton_model or r['parameters']['beam_seed']!=beam_seed or r['parameters']['mode']!=mode or (source=='synch' and (r['parameters']['electron_distribution']!=distribution or r['parameters']['synch_model']!=synch_model)):
    st.info('Pulse Calcular espectro para aplicar el modelo seleccionado.')
    st.stop()
if r['parameters']!=asdict(p):
    st.warning('Se muestra el último cálculo correcto; los parámetros enviados no coinciden con él.')
q=Parameters(**r['parameters'])
st.subheader(q.mode)
if q.compton:
    st.caption('Modelo: térmico Klein–Nishina' if q.compton_model=='thermal' else 'Modelo: RCS magnetizado local · Thomson resonante sin recoil')
    if 'rcs_density_cm3' in r:
        st.caption(f'Energía ciclotrón local: {r["rcs_EB_keV"]:.3g} keV · densidad de cargas derivada: {r["rcs_density_cm3"]:.3g} cm⁻³ · máxima energía en reposo en una colisión: {r["rcs_stats"]["max_rest_energy_keV"]:.3g} keV')
a,b,c,d=st.columns(4)
a.metric('τ resonante de referencia' if q.compton and q.compton_model=='magnetic' else 'τ Thomson radial',
         f'{q.rcs_tau:.3g}' if q.compton and q.compton_model=='magnetic' else (f'{r["tau"]:.3g}' if q.compton or q.source=='brems' else '—'))
b.metric('Luminosidad semilla',f'{r["seed_luminosity"]:.2e}',help='erg/s, integrada en la banda semilla elegida.')
c.metric('Energía salida / entrada',f'{r["energy_ratio"]:.3f} ×' if q.compton else '—',help='Incluye todas las energías transportadas, también fuera del gráfico.')
d.metric('Fotones con ganancia neta',f'{r["upscattered_fraction"]:.1%}' if q.compton else '—',help='Fracción ponderada por número de fotones. Se compara energía final con inicial.')

left,right=st.columns([2,1])
with left:
    ordinate=st.radio('Representación',['E · F_E','F_E'],horizontal=True,
                      help='E · F_E muestra la energía por intervalo logarítmico.',key='ordinate')
with right:
    show_parts=st.checkbox('Mostrar componentes',value=True)
normalize=False; instrument=0.
if q.source=='synch':
    normalize=st.checkbox('Normalizar a máximo del panel',value=False)
    instrument=st.slider('Resolución instrumental σ ln ν (solo visualización)',0.,.2,0.,.001,format='%.3f')
    st.caption('La resolución instrumental suaviza la vista; no modifica las semillas Compton ni los CSV físicos.')

COLORS={'intrinsic':'#74849a','emergent':'#152d48','direct':'#2b91a4','scattered':'#ee8c35','upscattered':'#bf4b70'}
def make_figure(zoom=False):
    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained')
    E=r['energy']; multiplier=E if ordinate=='E · F_E' else np.ones_like(E)
    def line(key,label,**kwargs):
        y=r[key]*multiplier
        ax.plot(E,np.where(y>0,y,np.nan),label=label,color=COLORS[key],**kwargs)
    if 'harmonics' in r:
        he=r['harmonic_energy']; hm=he if ordinate=='E · F_E' else np.ones_like(he)
        if show_parts and r['harmonics']:
            for order,values in r['harmonics'].items():
                ax.plot(he,np.where(values>0,values*hm,np.nan),lw=1.1,alpha=.8,label=f'n = {order}')
            remainder=np.maximum(0,r['harmonic_total']-sum(r['harmonics'].values()))
            ax.plot(he,np.where(remainder>0,remainder*hm,np.nan),color='gray',ls=':',label=f'n > {q.harmonic_show}')
        ax.plot(he,np.where(r['harmonic_total']>0,r['harmonic_total']*hm,np.nan),
                color='black',lw=2,ls='--' if q.compton else '-',label=f'Semilla: suma n ≤ {q.harmonic_max}' if q.synch_model=='harmonics' else 'Continuo sincrotrón')
    else:
        line('intrinsic','Semilla intrínseca' if q.compton else 'Emisión intrínseca',ls='--' if q.compton else '-',lw=2)
    if q.compton:
        line('emergent','Total emergente',lw=2.1)
        if show_parts:
            line('direct','Directa, sin dispersión',lw=1.2,alpha=.85)
            line('scattered','Dispersada (todos los fotones)',lw=1.4)
            line('upscattered','Dispersada con ganancia neta',lw=1.2,ls=':')
        lower=(r['emergent']-r['error'])*multiplier
        upper=(r['emergent']+r['error'])*multiplier
        if instrument==0:
            ax.fill_between(E,np.where(lower>0,lower,np.nan),upper,color=COLORS['emergent'],alpha=.13,label='±1σ Monte Carlo')
    if instrument>0:
        from scipy.ndimage import gaussian_filter1d
        for curve in ax.lines:
            xx,yy=curve.get_data(); pp=yy if ordinate=='E · F_E' else yy*xx
            smoothed=gaussian_filter1d(np.nan_to_num(pp),instrument/np.log(xx[1]/xx[0]),mode='constant')
            curve.set_ydata(np.where(smoothed>0,smoothed if ordinate=='E · F_E' else smoothed/xx,np.nan))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim((max(1,r['bins'][0]),min(100,r['bins'][-1])) if zoom else (r['bins'][0],r['bins'][-1]))
    visible=(E>=ax.get_xlim()[0])&(E<=ax.get_xlim()[1])
    curves=[r['intrinsic']*multiplier,r['emergent']*multiplier]
    positives=np.concatenate([y[visible & (y>0)] for y in curves])
    if 'harmonics' in r:
        hv=(he>=ax.get_xlim()[0])&(he<=ax.get_xlim()[1])
        hy=r['harmonic_total']*hm
        positives=np.concatenate([positives,hy[hv & (hy>0)]])
    if instrument>0:
        rendered=[]
        for curve in ax.lines:
            xx,yy=curve.get_data(); valid=(xx>=ax.get_xlim()[0])&(xx<=ax.get_xlim()[1])&np.isfinite(yy)&(yy>0)
            rendered.extend(yy[valid])
        positives=np.asarray(rendered)
    if len(positives):
        peak=max(positives); ax.set_ylim(max(min(positives)*.4,peak*1e-10),peak*3)
    else:
        ax.set_ylim(1e-30,1)
        ax.text(.5,.5,'Sin señal calculable en esta banda',ha='center',transform=ax.transAxes)
    if normalize and len(positives):
        scale=max(positives)
        for curve in ax.lines: curve.set_ydata(curve.get_ydata()/scale)
        for collection in ax.collections:
            for path in collection.get_paths(): path.vertices[:,1]/=scale
        ymin,ymax=ax.get_ylim(); ax.set_ylim(ymin/scale,ymax/scale)
    ax.set_xlabel('Energía del fotón [keV]')
    ax.set_ylabel('E · F_E [erg cm⁻² s⁻¹]' if ordinate=='E · F_E' else 'F_E [erg cm⁻² s⁻¹ keV⁻¹]')
    if normalize: ax.set_ylabel('Espectro / máximo del panel')
    ax.grid(True,which='major',alpha=.15); ax.legend(fontsize=9,ncol=2)
    ax.spines[['top','right']].set_visible(False)
    return fig

main_tab,zoom_tab,photon_tab,data_tab=st.tabs(['Espectro completo','Detalle 1–100 keV','Redistribución de fotones','Datos y descarga'])
with main_tab:
    fig=make_figure(); st.pyplot(fig,width='stretch')
    image=BytesIO(); fig.savefig(image,format='png',dpi=180); plt.close(fig)
    st.caption('Directa + dispersada = total emergente. La curva con ganancia neta es un subconjunto de la dispersada; no se suma otra vez.' if q.compton else
               'Emisión ópticamente delgada, antes de cualquier transporte Compton.')
with zoom_tab:
    if r['bins'][0]<100 and r['bins'][-1]>1:
        figzoom=make_figure(True); st.pyplot(figzoom,width='stretch'); plt.close(figzoom)
    else: st.info('La banda calculada no intersecta 1–100 keV. Amplíela y recalcule.')
with photon_tab:
    if 'seed_photons' in r:
        en=r['energy']
        figp,(axp,axd)=plt.subplots(2,1,figsize=(11,8),sharex=True,layout='constrained')
        for key,label,color in [('seed_photons','Semilla (mismos paquetes)','#74849a'),
                                 ('emergent_photons','Todos los fotones emergentes','#152d48'),
                                 ('upscattered_photons','Fotones con ganancia neta de energía','#bf4b70')]:
            yy=en*r[key]
            axp.plot(en,np.where(yy>0,yy,np.nan),label=label,color=color)
        diff=en*r['photon_excess']
        axd.plot(en,np.where(diff>0,diff,np.nan),color='#bf4b70',label='Exceso neto: fotones recibidos')
        axd.plot(en,np.where(diff<0,-diff,np.nan),color='#2b91a4',label='Déficit neto: fotones retirados')
        for aa in (axp,axd):
            aa.set_xscale('log'); aa.set_yscale('log'); aa.grid(alpha=.2)
            aa.set_ylabel('dN / dln E [fotones cm⁻² s⁻¹]'); aa.legend(fontsize=9)
            if q.source=='beam':
                aa.axvline(q.beam_max_keV,color='gray',ls='--',alpha=.6)
                if q.beam_max_keV<r['bins'][-1]: aa.axvspan(q.beam_max_keV,r['bins'][-1],color='#bf4b70',alpha=.05)
        axd.set_xlim(r['bins'][0],r['bins'][-1]); axd.set_xlabel('Energía del fotón [keV]')
        st.pyplot(figp,width='stretch'); plt.close(figp)
        st.caption('El panel inferior compara los mismos paquetes antes y después de Compton: muestra magnitudes positivas de excesos y déficits. Las fluctuaciones pequeñas pueden ser ruido Monte Carlo. La banda sombreada empieza en el máximo energético del haz semilla.')
        st.write(f'Máxima energía emergente muestreada: **{r["max_escaped_keV"]:.3g} keV**. '
                 f'Fotones por encima del máximo de la semilla: **{r["beyond_seed_fraction"]:.2%}**.')
    else:
        st.info('Esta comparación está disponible para modelos con comptonización y semilla no nula.')
with data_tab:
    keys=['energy','intrinsic','direct','scattered','upscattered','emergent','error']
    names=['E_keV','semilla_F_E','directa_F_E','dispersada_F_E','con_ganancia_F_E','emergente_F_E','error_MC_F_E']
    if 'seed_photons' in r:
        keys+=['seed_photons','emergent_photons','upscattered_photons','photon_excess']
        names+=['semilla_fotones_cm2_s_keV','emergente_fotones_cm2_s_keV','con_ganancia_fotones_cm2_s_keV','exceso_neto_fotones_cm2_s_keV']
    table={name:r[key] for name,key in zip(names,keys)}
    st.dataframe(table,width='stretch')
    csv=StringIO(); np.savetxt(csv,np.column_stack([r[k] for k in keys]),delimiter=',',header=','.join(names),comments='')
    if 'harmonics' in r:
        harmonic_csv=StringIO()
        orders=list(r['harmonics'])
        np.savetxt(harmonic_csv,np.column_stack([r['harmonic_energy'],r['harmonic_total']]+[r['harmonics'][n] for n in orders]),
                   delimiter=',',header='E_keV,total_F_E'+''.join(f',n_{n}_F_E' for n in orders),comments='')
        st.download_button('Descargar armónicos (malla fina)',harmonic_csv.getvalue(),'armonicos.csv','text/csv')
    x,y,z=st.columns(3)
    x.download_button('Descargar CSV',csv.getvalue(),'espectro.csv','text/csv')
    y.download_button('Descargar parámetros',json.dumps(r['parameters'],indent=2),'parametros.json','application/json')
    z.download_button('Descargar figura',image.getvalue(),'espectro.png','image/png')
    st.caption('Los flujos del CSV son F_E en erg cm⁻² s⁻¹ keV⁻¹. El error corresponde al espectro emergente.')

if q.compton:
    st.markdown('**Balance sobre todos los fotones transportados**')
    c1,c2,c3=st.columns(3)
    c1.metric('Intercambio con la corona',f'{r["energy_exchange"]:+.2e} erg/s',help='Positivo: los fotones reciben energía del plasma; negativo: la ceden.')
    c2.metric('Potencia fuera del gráfico',f'{r["outside_energy_fraction"]:.1%}')
    c3.metric('Luminosidad emergente ≥100 keV',f'{r.get("hard_luminosity",0):.2e} erg/s',help='Incluye también fotones que ya entraron con ≥100 keV.')
    if 'seed_hard_luminosity' in r:
        st.caption(f'Luminosidad semilla ≥100 keV: {r["seed_hard_luminosity"]:.2e} erg/s. '
                   f'Fracción dispersada: {r["scattered_fraction"]:.1%}.')
for message in r['messages']: st.info(message)

with st.expander('Supuestos físicos y cómo interpretar la comparación'):
    st.markdown('''
- **Bremsstrahlung:** hidrógeno ionizado, electrones Maxwell–Jüttner y sección eficaz electrón–ión Bethe–Heitler con corrección de Elwert. No incluye electrón–electrón ni absorción.
- **Compton térmico:** transporte espacial Monte Carlo en una esfera homogénea, electrones térmicos isotrópicos, transformaciones de Lorentz y Klein–Nishina. Las colisiones pueden aumentar o reducir la energía. La temperatura es impuesta; no se resuelve el equilibrio energético de la corona.
- **Compton magnetizado:** RCS local subcrítico, electrones 1-D en dos flujos opuestos con dispersión térmica paralela, resonancia ciclotrón y distribución angular 1+cos²θ en el reposo del electrón. Mezcla lognormal de magnitudes de B truncada a ±4σ. Dispersión coherente sin recoil; no se calcula polarización persistente ni una magnetosfera global. El transporte incluye colisiones múltiples y escape. Se exige B máximo local ≤0.03 BQ. τ resonante fija la densidad de la capa de dispersión, independiente de la población emisora.
- **Haz power law o cuerpo negro:** ilumina uniformemente el disco de la esfera. Se representa el flujo equivalente promediado angularmente, L_E/(4πd²), no una línea de visión del haz.
- **Ciclotrón–sincrotrón:** mismo motor del notebook, con elección de armónicos clásicos o continuo ultrarrelativista y electrones térmicos Maxwell–Jüttner o monoenergéticos, isotrópicos en dirección. El continuo sustituye la suma de armónicos. Se avisa cuando no es válida la aproximación ultrarrelativista. kT ensancha y desplaza los órdenes; σ ln B representa una mezcla de campos. Se usa el ángulo seleccionado y luminosidad equivalente isotrópica. La suma está truncada en nmax: aumentarlo para verificar convergencia.
- **Combinaciones:** se transportan los fotones semilla una sola vez. El total emergente ya incluye la componente directa y todas las dispersiones. No se suma de nuevo el espectro intrínseco.
- **Energías altas:** no se corta el transporte en 100 keV ni en los límites del gráfico. Se registra la potencia que escapa fuera de la banda. La cota del gráfico no es una cota física del cálculo.
- **Alcance:** sin absorción, pares, cascadas, reflexión, campos gravitatorios ni evolución temporal. A altas luminosidades o campos, estas omisiones pueden ser importantes.
- **Convergencia:** ampliar la banda semilla, aumentar paquetes y bins con cuidado, y comparar semillas aleatorias independientes. El muestreo logarítmico ponderado mejora la representación de las colas; no elimina la incertidumbre Monte Carlo.
''')
    st.markdown('Fuentes: [Bremsstrahlung relativista](https://arxiv.org/html/1911.08861), '
                '[ciclotrón y sincrotrón](https://arxiv.org/html/astro-ph/9601073), '
                '[Klein–Nishina](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html).')
with st.expander('Modelo resonante: referencias y normalización'):
    st.markdown('Se usa la sección eficaz de resonancia delta en el límite Thomson, promediada sobre la distribución local de campos. τ_ref = n_cargas σ_pico R. Es una implementación local propia, no una reproducción de los modelos globales publicados. Fuentes: [Nobili et al. 2008](https://doi.org/10.1111/j.1365-2966.2008.13125.x), [Yamasaki et al. 2020](https://doi.org/10.1093/mnras/staa2223). Para ajustes físicos de colas duras pueden ser necesarios recoil y secciones eficaces QED.')
with st.expander('Parámetros del cálculo mostrado'):
    st.json(r['parameters'])
