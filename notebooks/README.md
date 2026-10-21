# Espectros de un plasma relativista

Abra `plasma_relativista.ipynb` en JupyterLab o VS Code y ejecute todas las celdas. El notebook es autocontenido: no necesita importar `plasma_model.py`.

Instalación en un entorno Python:

```bash
python -m pip install -r ../requirements-dev.txt
jupyter lab plasma_relativista.ipynb
```

Edite los parámetros o use el panel y pulse «Calcular espectros». El tamaño corresponde al **radio** de una esfera homogénea de hidrógeno ionizado. Los espectros abarcan 1–100 keV. El notebook explica las unidades, normalización y aproximaciones; incluye referencias físicas y exportación a CSV/PNG.

`plasma_model.py` contiene una copia del motor para uso desde scripts. `test_plasma.py` verifica normalización Maxwell–Jüttner, transformaciones de Lorentz, recoil, límite Thomson, escape, escalados de brems, convergencia de cuadratura y límite no relativista. Ejecución: `python test_plasma.py`.

No es un modelo de equilibrio radiativo ni incluye brems electrón–electrón o absorción. El haz se integra sobre todas las direcciones: su flujo equivale al promedio angular a la distancia indicada. El corte inferior de las semillas brems es un parámetro explícito que debe someterse a un estudio de convergencia.
