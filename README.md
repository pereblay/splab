# Laboratorio de espectros

Demostrador interactivo en español de emisión de plasmas:

- Bremsstrahlung relativista.
- Comptonización térmica Klein–Nishina o magnetizada resonante local, con semilla de cuerpo negro predeterminada y power law opcional.
- Ciclotrón–sincrotrón con armónicos individuales y continuo ultrarrelativista opcional.

Incluye controles de parámetros, espectros y descarga de datos. Los modelos son demostradores: consulte las aproximaciones y límites en [la documentación científica](plasma_app/README.md) y [el modelo magnetizado](plasma_app/MAGNETIC_MODEL.md).

## Ejecutar localmente

Use Python **3.14** (validado con 3.14.8). Desde esta carpeta:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run plasma_app/app.py
```

En Windows, active el entorno con `.venv\Scripts\activate`.
Las versiones de las cuatro dependencias directas están fijadas a las utilizadas en las pruebas. No se necesitan credenciales ni servicios externos.

## GitHub y Streamlit Community Cloud

Suba **el contenido de esta carpeta** a la raíz de su repositorio, conservando `plasma_app/` y `notebooks/`. Los dos módulos científicos en `notebooks/` son necesarios para la aplicación.

Al crear la aplicación en Streamlit Community Cloud seleccione:

- Su repositorio y la rama que haya subido.
- **Main file path:** `plasma_app/app.py`.
- **Advanced settings → Python version:** `3.14`.

Las dependencias se instalan desde `requirements.txt` en la raíz. La versión de Python se elige en la configuración del despliegue. No hay que subir el entorno virtual. La publicación y la asociación al repositorio quedan a cargo del propietario.

Instrucciones oficiales: [desplegar una aplicación](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) y [dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

Los cálculos Monte Carlo y la suma de muchos armónicos pueden tardar en servidores compartidos. Empiece con los valores predeterminados y aumente la resolución progresivamente. El despliegue remoto aún no se ha realizado.

## Notebooks y pruebas

```bash
python -m pip install -r requirements-dev.txt
python -m pytest plasma_app notebooks -q
python -m jupyter lab notebooks
```

Los notebooks originales incluyen resultados guardados; sus opciones pueden diferir de las de la aplicación. Para ejecución local de la app use el comando anterior: no depende de rutas personales ni de un entorno virtual incluido en el repositorio.
