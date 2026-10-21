from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).with_name('app.py')

def test_modes_and_recalculation():
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    assert not at.exception
    assert at.selectbox(key='mode').options==['Bremsstrahlung','Comptonización','Sincrotrón']
    for mode in ['Bremsstrahlung','Sincrotrón','Comptonización']:
        at.selectbox(key='mode').set_value(mode).run()
        at.button[0].click().run()
        assert not at.exception,[e.message for e in at.exception]
        assert not at.error,[e.value for e in at.error]
        assert at.session_state['result']['parameters']['mode']==mode
    # La representación no recalcula ni modifica la física.
    original=at.session_state['result']['energy_ratio']
    at.radio(key='ordinate').set_value('F_E').run()
    assert not at.exception
    assert at.session_state['result']['energy_ratio']==original


def test_invalid_input_keeps_last_success():
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    original=at.session_state['result']['parameters'].copy()
    density=next(x for x in at.number_input if x.label=='Densidad térmica ne [cm⁻³]')
    density.set_value(1e23)
    at.button[0].click().run()
    assert not at.exception
    assert at.error
    assert at.session_state['result']['parameters']==original


def test_continuum_selector_and_mono():
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    at.selectbox(key='mode').set_value('Sincrotrón').run()
    at.selectbox(key='synch_model').set_value('continuum').run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert at.session_state['result']['parameters']['synch_model']=='continuum'
    assert at.session_state['result']['harmonics']=={}
    at.selectbox(key='synch_distribution').set_value('mono').run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert at.session_state['result']['parameters']['electron_distribution']=='mono'


def test_magnetic_selector_and_blackbody_defaults():
    at=AppTest.from_file(str(APP),default_timeout=120).run()
    assert not at.exception
    assert at.selectbox(key='seed_thermal').value=='blackbody'
    assert at.session_state['result']['parameters']['beam_seed']=='blackbody'
    at.selectbox(key='compton_model').set_value('magnetic').run()
    assert at.selectbox(key='seed_magnetic').value=='blackbody'
    at.button[0].click().run()
    assert not at.exception and not at.error
    r=at.session_state['result']
    assert r['parameters']['compton_model']=='magnetic' and 'rcs_stats' in r
    at.selectbox(key='seed_magnetic').set_value('powerlaw').run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert at.session_state['result']['parameters']['beam_seed']=='powerlaw'
