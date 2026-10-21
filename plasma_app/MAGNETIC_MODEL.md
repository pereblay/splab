# Compton térmico y magnetizado: qué se calcula

El selector de modelo térmico o magnetizado aparece en el proceso Comptonización.

## Térmico

Se conserva el motor Klein–Nishina para electrones libres isotrópicos Maxwell–Jüttner. La semilla externa es ahora **cuerpo negro por defecto**, kT=0.4 keV. Puede elegirse una power law. La luminosidad se integra en la banda semilla indicada; no se interpreta como luminosidad bolométrica fuera de esa banda.

## Magnetizado: aproximación RCS local

Es una implementación propia del límite Thomson resonante, **no** un modelo global de magnetar, una tabla NTZ ni un solver de Compton QED. Es adecuada para experimentar con resonancia, ángulos y cambios Doppler dentro de sus supuestos. No permite por sí sola ajustar físicamente colas duras observadas.

- Esfera homogénea de radio R y eje magnético fijo. Salida integrada sobre direcciones, representada como L_E/(4πd²).
- Dos contraflujos de igual densidad a lo largo de B, con gamma colectivo especificado. Cada flujo tiene una distribución térmica 1-D en su propio reposo, f(p_parallel) ∝ exp[-(sqrt(1+p²)-1)/theta]. Al muestrear en el laboratorio se incluye el jacobiano relativista.
- La magnitud local de B se representa por una mezcla lognormal truncada en ±4 sigma, con mediana B0. Esta mezcla microscópica es homogénea estadísticamente: no es un dipolo espacial ni un gradiente radial.
- Se usa la resonancia delta clásica en el reposo del electrón, sigma(omega,mu)=pi² r_e c (1+mu²) delta(omega-omega_B), promediada analíticamente sobre la distribución de B. En energía: sigma(E,mu)=C (1+mu²) p_EB(E), C=pi² r_e c hbar/keV_erg.
- La profundidad de referencia es **tau_ref=n_cargas sigma_pico R**. No es tau Thomson ni la profundidad real de cada fotón, que depende de energía, ángulo y velocidad. La app deriva y muestra la densidad de cargas.
- Transporte por colisiones nulas con cota 2 tau_ref, incluyendo el factor relativo 1-beta mu. Se siguen todas las dispersiones y escapes, conservando el número/peso de los fotones.
- La dispersión es coherente en el reposo electrónico; la energía cambia mediante transformaciones Doppler. La distribución angular de salida es proporcional a 1+mu_final² respecto a B. Se adopta un cierre no polarizado en cada colisión: no se conserva memoria de polarización.
- Se exige B0 exp(4 sigma) ≤ 0.03 BQ, con BQ=4.414×10¹³ G. Así, toda resonancia aceptada tiene E_rest ≤0.03 m_e c². **Es un límite sobre el campo local de la capa, no el campo superficial del magnetar.** Una energía alta en el laboratorio no equivale a una energía alta en el reposo de la carga.

Omisiones: recoil, secciones eficaces QED, polarización persistente, niveles de Landau altos, dispersión no resonante, pares/cascadas, campo dipolar/twist, gravedad, retroacción y equilibrio energético de los flujos. El campo que controla la semilla sincrotrón y el de la capa RCS son parámetros distintos.

## Referencias

Se toma la sección eficaz resonante clásica y la motivación de movimiento 1-D de:

- [Nobili, Turolla & Zane (2008)](https://doi.org/10.1111/j.1365-2966.2008.13125.x).
- [Yamasaki et al. (2020)](https://doi.org/10.1093/mnras/staa2223).

La geometría homogénea, mezcla de B y contraflujos de este demostrador son elecciones propias, no reproducciones de las geometrías de esos artículos. Para el tratamiento relativista magnético completo, véase [Nobili et al. (2008), II](https://academic.oup.com/mnras/article/389/2/989/974698).

## Pruebas

`test_magnetic.py` comprueba normalización de la resonancia, cota de rechazo, ley angular, coherencia en el marco electrónico, simetría de contraflujos, tau=0, transmisión analítica de una esfera casi estática, ganancia de energía, conservación del número de fotones, normalización blackbody y dominio de validez.
