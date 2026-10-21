# Ciclotrón y sincrotrón: armónicos y continuo

Abra `ciclotron_sincrotron.ipynb` y ejecute todas las celdas. Es autocontenido. Usa las mismas dependencias de `../requirements-dev.txt`.

El panel permite editar campo magnético, temperatura o energía monoenergética, ángulo de observación, dispersión del campo, resolución instrumental y número de armónicos. Muestra dos gráficos: radio (banda editable) y 1–100 keV. Por defecto son **dos escenarios con campos distintos**; puede activar el mismo campo en ambos.

Los primeros órdenes aparecen individualmente, junto con el resto y la suma. Hay preajustes para armónicos separados, solapamiento térmico, campo inhomogéneo y continuo sincrotrón. El modo continuo es ultrarrelativista y no se suma a los armónicos, porque describen el mismo proceso.

La salida física es potencia por electrón y por estereorradián, por intervalo logarítmico. Puede normalizar cada vista a su máximo. La exportación incluye CSV, PNG y parámetros JSON; los CSV siempre mantienen unidades físicas.

El cálculo es clásico y ópticamente delgado. No incluye absorción, efectos de plasma ni ciclotrón cuántico. El notebook documenta estas limitaciones y muestra avisos de validez y truncamiento.

`ciclosincrotron_model.py` es la copia del motor para scripts. Verificaciones: `python test_ciclosincrotron.py`.
