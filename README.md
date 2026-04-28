# PLI Demo | Predictive Location Intelligence by BDS

Demo funcional en Streamlit para vender la práctica **Predictive Location Intelligence (PLI)** de Business Data Scientists.

## Objetivo

Mostrar a un cliente cómo PLI convierte datos de tiendas, candidatos de apertura y variables geoespaciales en decisiones económicas:

- dónde abrir,
- dónde no invertir,
- dónde hay riesgo de canibalización,
- qué portafolio maximiza retorno,
- cómo convertir el análisis en un memo ejecutivo.

## Cómo correrlo

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecuta la app:

```bash
streamlit run app.py
```

3. Abre el navegador en la URL local que indique Streamlit.

## Qué contiene

La maqueta incluye datos sintéticos, modelo predictivo, mapa, ranking de ubicaciones, business case, análisis de canibalización, market hold capacity, backtesting y memo ejecutivo.

## Cómo adaptarlo a un cliente real

Reemplaza la generación sintética por un POSAR real con:

1. ventas y transacciones,
2. ubicación y trade area,
3. demografía y economía local,
4. competencia,
5. operación de tienda,
6. inmueble y CAPEX,
7. historial de decisiones.

## Notas

El demo no usa APIs externas. La versión real puede integrarse con fuentes de datos internas, INEGI, POIs, movilidad, competencia, CRM, ERP, WMS, POS y dashboards de Power BI.
