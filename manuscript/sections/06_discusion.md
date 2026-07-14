# 5. Discusión

## 5.1 Unificación de Investigación de Operaciones y Deep Learning

Este trabajo demuestra que la orquestación hospitalaria en urgencias no requiere fusionar indiscriminadamente metodologías predictivas y logísticas, sino articularlas en una arquitectura ciberfísica donde cada disciplina conserva identidad y aporta capacidades complementarias. La Ingeniería Industrial provee el gemelo digital SimPy, restricciones de capacidad y métricas operativas (LOS, ocupación, equidad); el Deep Learning provee optimización secuencial de políticas de despacho mediante DQN; la Informática Biomédica provee XAI y diseño de interfaces para adopción clínica.

Este enfoque trasciende frameworks MLOps estáticos como el de Soster et al. (2025), donde predicciones ML alimentan programación lineal offline. Aquí, el agente DRL interactúa continuamente con un gemelo digital sincronizado, habilitando orquestación adaptativa en horizontes de minutos — no solo planificación de turnos en horizontes de horas.

## 5.2 Novedad de la orquestación adaptativa en tiempo real

Los métodos reactivos (ESI+FIFO, SRPT) aplican reglas fijas independientemente del contexto global del departamento. El DQN+DT selecciona entre políticas compuestas según estado s ∈ R^17, incorporando señales operativas y fisiológicas agregadas. La sincronización bidireccional permite anticipar saturación mediante proyección estocástica antes de manifestación clínica del colapso — cerrando la **Brecha 1** (simulación pasiva vs. gemelo activo) y la **Brecha 2** (predicción vs. orquestación DRL).

## 5.3 Cierre de la brecha XAI–Factores Humanos

La capa SHAP con paneles diferenciados aborda la **Brecha 3**: no basta prescribir una acción; es indispensable comunicar covariables determinantes en lenguaje operativo comprensible. La transparencia algorítmica se concibe como requisito de confianza, no como feature cosmético — coherente con el escepticismo clínico documentado hacia automatización en contextos críticos (Adel et al., 2025).

## 5.4 Limitaciones

**Complejidad computacional:** Los mini-runs SimPy en look-ahead tienen costo O(N·T); en producción se balancea precisión vs. latencia mediante fast_cap y horizon configurables.

**Calidad de datos:** RTLS y EHR requieren pipelines de integración hospitalaria con latencia y completitud variables. La validación actual utiliza parámetros calibrados de literatura; recalibración local es mandatoria.

**Resistencia clínica inicial:** El modo shadow (recomendación sin ejecución automática) mitiga riesgo pero extiende timeline de adopción.

**Equidad en simulación:** El índice de equidad reportó valores nulos en baselines cuando las esperas por ESI fueron homogéneamente bajas en triaje inmediato — fenómeno que requiere calibración adicional del modelo de colas.

## 5.5 Posicionamiento frente a literatura Q1/Q2

| Trabajo | Contribución | Limitación vs. propuesta |
|---|---|---|
| Liu et al. (2020) | DRL en ED | Sin gemelo bidireccional ni XAI |
| Soster et al. (2025) | MLOps + PL | Planificación offline, no orquestación continua |
| Vural et al. (2025) | Predicción hacinamiento XAI | Sin despacho DRL |
| Applied Sciences (2024) | DES + ML scheduling | Simulación retrospectiva |
| Hodgson et al. (2025) | Flujo vertical IA | Sin MDP ni gemelo activo |

## 5.6 Generalización

La arquitectura modular (Digital Twin + DRL + XAI) es transferible a UCI (orquestación de camas críticas), quirófanos (programación estocástica de bloques) y farmacia hospitalaria (despacho de medicamentos time-sensitive), mediante recalibración de topología SimPy, espacio de estados MDP y función de recompensa.
