# 3. Metodología

## 3.1 Infraestructura del Digital Twin Activo

### 3.1.1 Arquitectura del gemelo digital

El gemelo digital se implementó en Python 3.11+ con SimPy 4.1. La topología replica el flujo clínico-estándar del ED: llegada → espera → triaje → consulta → imagenología (opcional) → disposición → egreso/admisión. Recursos modelados: 25 camas, 4 consultorios, 2 salas de imagenología, 2 estaciones de triaje (calibrados según literatura; Applied Sciences, 2024).

**Fuentes de datos:** En validación computacional, los parámetros estocásticos se calibraron sobre distribuciones documentadas. En despliegue productivo, el State Manager ingiere telemetría vía REST/WebSocket desde RTLS (posición) y EHR (ESI, vitals, órdenes).

### 3.1.2 Sincronización bidireccional

La clase `BidirectionalSync` implementa ciclos de sincronización cada 30 s (configurable). En cada ciclo:

1. `pull_from_physical()`: ingesta snapshot del estado ED (colas ESI, esperas, ocupación).
2. `run_stochastic_scenarios()`: ejecuta hasta 15 mini-runs SimPy de 30 min (inline) o 50 runs de 120 min (`run_lookahead()` en producción).
3. `push_to_virtual()`: actualiza proyección de ocupación y calcula divergencia.
4. `feedback_directive()`: emite recalibración si divergencia > ε = 0.05.

### 3.1.3 Distribuciones estocásticas

- **Llegadas:** Proceso de Poisson no homogéneo (NHPP) con tasa base 12 pacientes/hora y pico 2.5× entre 10:00–14:00.
- **ESI:** Multinomial calibrada (Green, 2006): ESI-1 2%, ESI-2 8%, ESI-3 25%, ESI-4 40%, ESI-5 25%.
- **Tiempos de servicio:** Lognormal por estación con multiplicadores ESI (triaje μ=8 min, consulta μ=45 min, imagenología μ=30 min, disposición μ=15 min).

### 3.1.4 Validación de fidelidad

Se aplicó test de Kolmogorov-Smirnov (KS) sobre tiempos de servicio simulados vs. referencia (n=5000) y MAPE sobre conteos horarios de llegada. Criterios: p_KS > 0.05 para estaciones; MAPE documentado para transparencia metodológica.

## 3.2 Formulación del Agente DRL-DQN

### 3.2.1 Espacio de estados (dimensión 17)

| Índice | Feature | Normalización |
|---|---|---|
| 0–4 | Cola ESI 1–5 | /10 |
| 5–9 | Espera ESI 1–5 | min/120, cap 1 |
| 10–12 | Ocupación camas/consulta/imagen | /capacidad |
| 13 | Hora del día | /24 |
| 14 | Pacientes en sistema | /camas |
| 15–16 | SpO2, FC promedio | /100, /200 |

### 3.2.2 Espacio de acciones (dimensión 6)

Cada acción selecciona una política de despacho compuesta sobre la cola de espera, actualizada cada segmento de episodio durante entrenamiento y cada ciclo de sync en inferencia.

### 3.2.3 Hiperparámetros de entrenamiento

| Parámetro | Valor |
|---|---|
| γ | 0.99 |
| Learning rate | 10^−4 |
| Batch size | 64 |
| Replay buffer | 50.000 |
| Target update | cada 100 pasos |
| ε inicial/final | 1.0 / 0.05 |
| Episodios | 150 (offline) |
| Episodio simulado | 8 horas |
| Réplicas Monte Carlo | 10 |

### 3.2.4 Estrategia ε-greedy

Durante entrenamiento offline, ε-greedy balancea exploración de políticas de despacho vs. explotación de Q-values aprendidos. En inferencia clínica, ε = 0 (greedy puro). La exploración en entrenamiento previene convergencia prematura a políticas subóptimas locales — crítico dado que el espacio de acciones compuesto es no convexo.

## 3.3 Diseño de la Interfaz XAI

### 3.3.1 Mecanismo de traducción

Tras selección de acción a* = argmax_a Q(s,a), SHAP KernelExplainer calcula atribuciones φ_i para las features de estado. Las cinco features con mayor |φ_i| alimentan plantillas narrativas en español que distinguen covariables operativas (colas, ocupación) de fisiológicas (SpO2, FC).

### 3.3.2 Visualización por rol

- **Panel clínico (Streamlit):** colas ESI en tiempo real, recomendación con justificación, alertas de saturación proyectada.
- **Panel gestor:** KPIs LOS/ocupación/equidad, trazas de sync del gemelo, logs de shadow mode.

### 3.3.3 Protocolo de validación con stakeholders

Se diseñó checklist cualitativo para revisión clínica: (1) coherencia clínica de la acción recomendada; (2) correspondencia entre atribuciones SHAP y juicio experto; (3) utilidad percibida de la narrativa; (4) disposición a confiar en modo shadow. Validación prospectiva in vivo queda como trabajo futuro.

## 3.4 Baselines y métricas

**Baselines:** ESI+FIFO (estándar clínico), SRPT (Shortest Remaining Processing Time), heurístico compuesto (ESI + probabilidad admisión + tiempo espera), DQN+DT (propuesta).

**Métricas:** LOS medio y P95, ocupación máxima, espera en casos críticos (ESI 1–2), índice de equidad (1 − Gini de esperas por ESI), throughput. Intervalos de confianza 95% por bootstrap (1000 remuestreos).
