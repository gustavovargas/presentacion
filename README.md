# Interpretabilidad, Machine Learning y Riesgo de Crédito

Trabajo Fin de Máster del [*máster en Data Science y Big Data en Finanzas*](https://www.afiescueladefinanzas.es/master-big-data-finanzas) de AFI Escuela de Finanzas.
Agosto 2019.

Los datos usados se pueden encontrar [aquí](https://drive.google.com/drive/folders/1YnLc_n87TREgyr8BeZ-vSwv8slX8hZpQ?usp=sharing).

La presentación del TFM se puede ver en este [link](https://gustavovargas.github.io/presentacion_tfm/). Está hecho en reveal.js.


## Abstract

En riesgo de crédito, un parámetro típico a calcular es la Probabilidad de Incumplimiento (_Probability of Default_), que indica la probabilidad de que un agente tenga algún problema futuro con el pago de su deuda.Típicamente esto se ha hecho con modelos estadísticos simples, como la regresión logística o la regresión lineal, ya que uno de los requisitos que pide el regulador es que estos sistemas de calificación se puedan explicar a terceros, y es que esos modelos son fácilmente _interpretables_, por lo que se puede confiar en ellosen la toma de decisiones.

Pero, a la postre, esto no deja de ser un problema de predicción, y a la hora de predecir los modelos de Machine Learning tienen un mejor rendimiento. El problema radica en que estos modelos suelen ser _cajas negras_, esto es, no se puede explicar cómo funcionan exactamente por dentro para un caso determinado. Enel presente trabajo aplicaremos los paquetes de interpretabilidad para modelos de Machine Learning que sehan estado construyendo en los últimos años para por hacer interpretables esascajas negas.

En este trabajo usamos **eli5**, **Lime** y **shap** para conseguir explicaciones de nuestros modelos aplicadossobre la base de datos que proporciona la empresa de préstamos LendingClub

## Referencias

