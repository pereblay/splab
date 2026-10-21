# Laboratorio de espectros — Streamlit

Aplicación local con cinco procesos:

- Bremsstrahlung.
- Comptonización de una semilla de cuerpo negro (predeterminada) o power law, con modelo térmico o magnetizado resonante.
- Ciclotrón–sincrotrón térmico, con armónicos individuales en altas energías.

## Arranque

Desde la raíz del proyecto, si el entorno ya está instalado:

```bash
plasma_app/.venv/bin/python -m streamlit run plasma_app/app.py --server.address 127.0.0.1
```

Para instalar en otro equipo (Python 3.11 o posterior):

```bash
python -m venv plasma_app/.venv
plasma_app/.venv/bin/python -m pip install -r plasma_app/requirements.txt
plasma_app/.venv/bin/python -m streamlit run plasma_app/app.py --server.address 127.0.0.1
```

En Windows, use `plasma_app\.venv\Scripts\python.exe`. Conserve también la carpeta `notebooks`: la aplicación reutiliza `plasma_model.py` y `ciclosincrotron_model.py`.

## Uso

Seleccione el proceso, cambie los parámetros de la barra lateral y pulse **Calcular espectro**. El cambio de representación F_E / E·F_E no vuelve a simular. La pestaña principal muestra por defecto 0.1–10000 keV; hay otra vista de 1–100 keV. La banda visible es editable.

La comptonización muestra la semilla, la componente directa, toda la componente dispersada y el subconjunto dispersado con ganancia neta. El total emergente es **directa + dispersada**. La semilla no se suma otra vez. El transporte no se detiene en los límites del gráfico.

Se informa del intercambio energético con la corona, del cociente energía de salida/entrada, de la luminosidad ≥100 keV y de la potencia fuera de la banda visible. Estas magnitudes incluyen todos los paquetes transportados. El muestreo de importancia logarítmico facilita la resolución de colas; cada paquete tiene un peso de número de fotones, conservado en las dispersiones.

Los resultados se pueden descargar como CSV, PNG y JSON de parámetros. El CSV contiene F_E en erg cm⁻² s⁻¹ keV⁻¹, independientemente de la representación elegida.

## Física y límites

- Esfera homogénea. `ne` fija tau radial y normaliza bremsstrahlung.
- Brems: Maxwell–Jüttner + Bethe–Heitler–Elwert para H, solo electrón–ión.
- Compton: Klein–Nishina, Lorentz y transporte espacial con múltiples dispersiones; kT=1–300 keV, tau≤3 por coste. Temperatura fija, sin balance térmico.
- Ciclotrón–sincrotrón: mismo motor de armónicos clásicos del notebook, distribución Maxwell–Jüttner isotrópica. Controles de kT (1–1000 keV), B, ángulo de observación, dispersión lognormal del campo, orden máximo y órdenes visibles. La densidad emisora normaliza el espectro. Por defecto kT=20 keV y B=10¹¹ G. Pruebe kT=50 y 100 keV con el mismo campo para observar ensanchamiento y solapamiento. Los armónicos se dibujan en una malla fina independiente de los bins Monte Carlo y se exportan en un CSV separado.
- Sincrotrón solo: flujo direccional; la luminosidad indicada es equivalente isotrópica en esa dirección. Compruebe convergencia en nmax y cuadratura angular; no se incluyen efectos cuánticos de Landau.
- Haz: flujo equivalente promediado angularmente, L_E/(4πd²). Fuentes internas: emisión isotrópica promediada.
- No hay autoabsorción, pares, cascadas, reflexión, enfriamiento ni efectos gravitatorios. Los fotones gamma se siguen con Klein–Nishina, pero la física de pares no está incluida.
- La banda semilla interna es un corte explícito; amplíela para estudiar convergencia. El flujo puede ser cero fuera del soporte calculado, lo cual no prueba ausencia física de emisión.

## Verificación

```bash
plasma_app/.venv/bin/python -m pip install pytest
plasma_app/.venv/bin/python -m pytest plasma_app/test_physics.py plasma_app/test_app.py -q
```

Los notebooks originales conservan sus propias pruebas físicas. Las pruebas de esta app comprueban los cinco procesos, tau=0, ganancia por encima del corte semilla, independencia del transporte respecto a la banda dibujada, normalización del sincrotrón, muestreo ponderado y validación de entradas.

## Opciones compartidas con el notebook

En Sincrotrón puede elegir **Armónicos clásicos** o **Continuo sincrotrón (ultrarrelativista)**, con electrones **Maxwell–Jüttner** o **monoenergéticos**. La energía monoenergética es cinética, en keV. El continuo sustituye los armónicos, no se añade a ellos; se muestran avisos si la energía electrónica es demasiado baja para esa aproximación. Sus valores iniciales son kT=5000 keV y B=10⁹ G.

Se mantienen el campo, el ángulo (emisión sin Compton), la dispersión lognormal de B, nmax, órdenes visibles, nodos de cuadratura y puntos de malla. La normalización y la resolución instrumental son controles de **visualización**: no modifican los fotones semilla, las métricas ni los CSV físicos. Con suavizado instrumental se omite la banda de incertidumbre Monte Carlo sin convolucionar. La app continúa centrada en altas energías; no añade el segundo escenario radio del notebook.

## Redistribución Compton y fotones de alta energía

La opción «Ampliar la banda para incluir todos los fotones simulados» extiende los extremos de la banda después del transporte, sin cambiar la física ni descartar paquetes. La pestaña «Redistribución de fotones» muestra dN/dln E de la semilla y de la salida, el subconjunto con ganancia neta de energía y los excesos/déficits netos por bin. Se usan los mismos paquetes ponderados antes y después: al incluir toda la banda, la integral del exceso neto de número de fotones es cero salvo redondeo.

Para un haz power law se marca su energía máxima; la región superior permite identificar los fotones desplazados más allá de ese corte. Se informa de su fracción y de la máxima energía emergente **muestreada** (no un corte físico). El CSV añade columnas de número de fotones por cm², segundo y keV.

## Selector térmico / magnetizado

El modo Comptonización permite seleccionar **Térmico · Klein–Nishina** o **Magnetizado · RCS Thomson local**. Ambos usan cuerpo negro por defecto para la semilla externa, con kT editable; la power law sigue disponible. Consulte [MAGNETIC_MODEL.md](MAGNETIC_MODEL.md) para ecuaciones, parámetros, normalización y límites: el modo RCS es un demostrador local subcrítico y no sustituye un modelo QED global para ajustar magnetares.
