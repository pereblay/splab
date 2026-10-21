from dataclasses import replace
import numpy as np
from scipy.integrate import trapezoid
from physics import *


def test_available_modes():
    for mode in MODES:
        p=Parameters(mode=mode,packets=10000,seed_grid=180)
        r=calculate(p)
        for name in ['intrinsic','emergent','direct','scattered','error']:
            assert np.all(np.isfinite(r[name])) and np.all(r[name]>=0),(mode,name)
        assert np.allclose(r['direct']+r['scattered'],r['emergent'],rtol=1e-12,atol=1e-40)
        if p.compton: assert r['packet_number_ratio']==1.


def test_zero_depth_identity():
    p=Parameters(ne_cm3=0.,packets=20000)
    r=calculate(p)
    assert np.array_equal(r['emergent'],r['seed_mc'])
    assert np.all(r['scattered']==0) and r['energy_ratio']==1.


def test_upscatter_beyond_incident_cutoff():
    # Fotones inicialmente <=1 keV deben aparecer a energías mayores.
    p=Parameters(kT_keV=150.,ne_cm3=2e17,beam_min_keV=.2,beam_max_keV=1.,packets=40000,
                 plot_min_keV=.01,plot_max_keV=1e4)
    r=calculate(p)
    assert np.any(r['upscattered'][r['energy']>10]>0)
    assert r['energy_ratio']>1.5
    assert r['energy_exchange']>0


def test_broadband_transport_independent_of_view():
    p=Parameters(packets=12000)
    r=calculate(p); other=calculate(replace(p,plot_max_keV=100.))
    assert r['energy_ratio']==other['energy_ratio']
    assert r['hard_luminosity']==other['hard_luminosity']
    assert r['outside_energy_fraction']<=other['outside_energy_fraction']


def test_synch_power_normalization():
    p=Parameters(mode='Sincrotrón',electron_distribution='mono',gamma_min=100.,B_G=1e3)
    E=np.geomspace(1e-12,1e4,5000)
    Q=synch_photon_luminosity(E,p)
    power=trapezoid(E*Q,E)*plasma.KEV
    volume=4*np.pi/3*(p.radius_km*1e5)**3
    expected=4*synch.E_CHARGE**4/(9*synch.M_E**2*synch.C**3)*p.B_G**2*p.gamma_min**2*p.nrel_cm3*volume
    assert abs(power/expected-1)<.001


def test_importance_normalization():
    p=Parameters(packets=200000,beam_seed='powerlaw')
    grid,Q=seed_spectrum(p)
    E,rate=importance_packets(grid,Q,p.packets,np.random.default_rng(42))
    assert abs(np.sum(E*rate)*plasma.KEV/p.beam_luminosity-1)<1e-4 # Γ=2: énergie/log constante.


def test_invalid_parameters():
    import pytest
    for p in [Parameters(ne_cm3=1e23),Parameters(plot_max_keV=.01),
              Parameters(mode='Sincrotrón',sigma_ln_B=-1)]:
        with pytest.raises(ValueError): p.validate()


def test_harmonics_match_notebook():
    p=Parameters(mode='Sincrotrón',harmonic_max=30,seed_min_keV=.5,seed_max_keV=30.)
    r=calculate(p)
    ref=synch.compute_band(p.B_G,.5*plasma.KEV/synch.H,30*plasma.KEV/synch.H,
        p.theta_deg,p.synch_kT_keV,'termica',50.,nmax=30,show_orders=8,
        sigma_ln_instrument=0.,points=1800)
    volume=4*np.pi/3*(p.radius_km*1e5)**3
    expected=ref['total']*p.nrel_cm3*volume/(p.distance_kpc*plasma.KPC)**2/r['harmonic_energy']
    assert np.allclose(r['harmonic_total'],expected,rtol=1e-12,atol=1e-50)
    assert len(r['harmonics'])==8
    hot=calculate(replace(p,synch_kT_keV=100.))
    assert not np.allclose(hot['harmonic_total']/hot['harmonic_total'].max(),
                           r['harmonic_total']/r['harmonic_total'].max())


def test_interpolation_preserves_harmonic_gaps():
    grid=np.array([1.,2.,3.,4.,5.]); q=np.array([1.,0.,0.,0.,1.])
    assert interpolate_photons(np.array([2.5,3.5]),grid,q).sum()==0


def test_continuum_and_mono_match_notebook():
    for model,dist,temp,kinetic,B in [('continuum','thermal',5000.,50.,1e9),
                                    ('continuum','mono',20.,10000.,1e9),
                                    ('harmonics','mono',20.,50.,1e11)]:
        p=Parameters(mode='Sincrotrón',synch_model=model,electron_distribution=dist,
                     synch_kT_keV=temp,kinetic_keV=kinetic,B_G=B,
                     seed_min_keV=.5,seed_max_keV=100.,seed_grid=400,harmonic_max=30)
        r=calculate(p)
        ref=synch.compute_band(B,.5*plasma.KEV/synch.H,100*plasma.KEV/synch.H,75.,temp,
            'termica' if dist=='thermal' else 'monoenergetica',kinetic,
            mode='sincrotron' if model=='continuum' else 'armonicos',
            nmax=30,show_orders=8,sigma_ln_instrument=0.,points=400)
        volume=4*np.pi/3*(p.radius_km*1e5)**3
        expected=ref['total']*p.nrel_cm3*volume/(p.distance_kpc*plasma.KPC)**2/r['harmonic_energy']
        assert np.allclose(r['harmonic_total'],expected,rtol=1e-12,atol=1e-50)
        if model=='continuum': assert r['harmonics']=={}


def test_continuum_regime_warning():
    p=Parameters(mode='Sincrotrón',synch_model='continuum',synch_kT_keV=20.,seed_grid=300)
    assert any('fuera del régimen' in m for m in calculate(p)['messages'])


def test_auto_range_and_photon_balance():
    p=Parameters(kT_keV=150.,ne_cm3=2e17,beam_min_keV=.2,beam_max_keV=1.,
                 plot_min_keV=.2,plot_max_keV=1.,packets=20000)
    r=calculate(p)
    assert r['bins'][-1]>10
    assert r['bins'][-1]>r['max_escaped_keV']
    assert r['outside_energy_fraction']==0
    widths=np.diff(r['bins'])
    assert abs(np.sum(r['photon_excess']*widths)) < np.sum(r['seed_photons']*widths)*1e-12
    assert np.any(r['emergent_photons'][r['energy']>10]>0)
    assert r['beyond_seed_fraction']>0
