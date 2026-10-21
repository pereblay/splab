import numpy as np
from scipy.integrate import trapezoid
from dataclasses import replace
from physics import Parameters, calculate, seed_spectrum, plasma
import magnetic_compton as m


def test_resonance_normalization_and_bound():
    B=3e11; sigma=.2; center=m.MEC2*B/m.BQ
    grid=np.geomspace(center*np.exp(-.9),center*np.exp(.9),100000)
    assert abs(trapezoid(m.field_pdf(grid,B,sigma),grid)-1)<2e-6
    assert np.max(m.resonance_ratio(grid,np.ones(len(grid)),B,sigma))<=1+1e-12
    assert np.all(m.resonance_ratio(np.array([center*.1,center*10]),np.ones(2),B,sigma)==0)


def test_rest_scattering_and_angular_law():
    rng=np.random.default_rng(17); N=100000
    E=np.full(N,3.); direction=np.tile([0.,0.,1.],(N,1))
    out,n,rest=m.scatter(E,direction,np.zeros(N),rng)
    assert np.array_equal(out,E)
    assert np.allclose(np.sum(n*n,axis=1),1,atol=1e-13)
    assert abs(np.mean(n[:,2]**2)-.4)<.004
    beta=np.full(N,-.98)
    out,n,rest=m.scatter(E,direction,beta,rng)
    transformed=out/np.sqrt(1-beta**2)*(1-beta*n[:,2])
    assert np.allclose(rest,transformed,rtol=1e-12)
    assert out.mean()>E.mean()


def test_stream_symmetry_and_zero_tau():
    rng=np.random.default_rng(33)
    beta=m.sample_parallel_beta(100000,5,10,rng)
    assert np.all(np.abs(beta)<1) and abs(beta.mean())<.01
    E=np.geomspace(.1,100,3000)
    out,counts,stats=m.transport(E,3e11,.2,0,5,10,rng)
    assert np.array_equal(out,E) and counts.sum()==0


def test_static_resonance_transmission():
    # At rest, broadening from kT=.01 is tiny compared to sigma_B=.2.
    # For a parallel uniform beam through a sphere:
    # <exp(-tau chord)> = 2*[1-(1+a)exp(-a)]/a², a=2*tau_eff.
    rng=np.random.default_rng(65); B=3e11; sigma=.2; tau=.4
    energy=m.MEC2*B/m.BQ*np.exp(-sigma*sigma)
    out,counts,_=m.transport(np.full(60000,energy),B,sigma,tau,.01,1,rng)
    a=2*tau
    expected=2*(1-(1+a)*np.exp(-a))/a**2
    assert abs(np.mean(counts==0)-expected)<.015
    assert len(out)==60000 and np.all(out>0)


def test_magnetic_tail_and_full_photon_balance():
    p=Parameters(compton_model='magnetic',beam_seed='blackbody',packets=20000,seed_grid=400)
    r=calculate(p)
    assert r['scattered_fraction']>0
    assert np.any(r['upscattered_photons'][r['energy']>10]>0)
    assert r['energy_ratio']>1
    assert r['rcs_stats']['max_rest_energy_keV']<=m.MEC2*.03
    assert np.isclose(np.sum(r['seed_photons']*np.diff(r['bins'])),
                      np.sum(r['emergent_photons']*np.diff(r['bins'])),rtol=1e-12)
    assert r['parameters']['compton_model']=='magnetic'


def test_blackbody_default_and_normalization():
    p=Parameters()
    assert p.beam_seed=='blackbody'
    grid,Q=seed_spectrum(p)
    assert np.isclose(trapezoid(grid*Q,grid)*plasma.KEV,p.beam_luminosity)
    # E² Q_E peaks at x=3.92069 for a blackbody per log energy interval.
    assert abs(grid[np.argmax(grid**2*Q)]/p.seed_kT_keV-3.92069)<.03
    r=calculate(replace(p,packets=10000))
    assert r['parameters']['beam_seed']=='blackbody' and r['scattered_fraction']>0


def test_qed_domain_guard():
    import pytest
    with pytest.raises(ValueError): m.validate(1e14,.2,3,5,10)


