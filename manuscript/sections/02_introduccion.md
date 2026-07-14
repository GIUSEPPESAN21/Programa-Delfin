# 1. Introducción

## 1.1 Contextualización de la crisis operativa en urgencias

Los departamentos de urgencias (Emergency Departments, ED) constituyen el nodo más crítico y congestionado de la cadena asistencial hospitalaria. Desde la perspectiva de la Ingeniería Industrial, operan como sistemas estocásticos donde convergen demanda impredecible, recursos finitos (camas, consultorios, personal clínico) y restricciones de tiempo biológico que no admiten postergación indefinida. El hacinamiento (overcrowding) se ha convertido en un indicador sistémico de falla operativa, asociado con incremento de mortalidad, demoras en tratamiento de condiciones time-sensitive y deterioro de la experiencia del paciente (Asplin et al., 2003; Hoot & Aronsky, 2008).

Históricamente, la gestión operativa se ha apoyado en metodologías de triaje manual —notablemente el Emergency Severity Index (ESI)— y en modelos analíticos de colas que asumen llegadas Poisson y tiempos de servicio exponenciales. Si bien el ESI proporciona una clasificación estandarizada de acuity, revisiones sistemáticas documentan variabilidad inter-observador, sensibilidad limitada para predecir ingreso a unidades críticas y escasa capacidad de anticipar cuellos de botella logísticos antes de su manifestación clínica (Adel et al., 2025; Farrohknia et al., 2011). Paralelamente, la teoría de colas clásica, aunque conceptualmente elegante, rara vez captura la heterogeneidad de flujos clínicos (fast-track, imagenología, admisión hospitalaria) ni la no estacionariedad de la demanda intradiaria.

La convergencia reciente entre Inteligencia Artificial (IA) e Ingeniería Industrial ha producido avances significativos en predicción clínica (Random Forest, Deep Learning, Transformers) y optimización logística (DES, programación lineal, gemelos digitales). Sin embargo, la literatura tiende a tratar estas líneas como pipelines secuenciales desacoplados: modelos predictivos que informan decisiones humanas sin retroalimentación operativa automatizada, o simulaciones retrospectivas que evalúan políticas sin sincronización con el plano físico en tiempo real (Álvarez-Vázquez et al., 2025; Applied Sciences, 2024).

## 1.2 Revisión crítica del estado del arte

**Triaje manual y escalas heurísticas.** El ESI permanece como estándar de facto en numerosos sistemas de salud. No obstante, su utilidad predictiva para outcomes operativos (LOS, abandono, reingreso) es moderada comparada con modelos de ML entrenados sobre EHR (Trotzky et al., 2026). La IA de triaje supera consistentemente escalas manuales en predicción de hospitalización y mortalidad, pero rara vez se integra con motores de despacho dinámico (Adel et al., 2025).

**Teoría de colas y programación.** Hodgson et al. (2025) demuestran que la reconfiguración espacial del ED —procesamiento vertical guiado por Random Forest— reduce demoras mediante segregación dinámica de flujos. Soster et al. (2025) integran predicciones ML con programación lineal entera para optimización de turnos, logrando incrementos sustanciales de capacidad en picos de demanda. Estos trabajos confirman la sinergia IA–Investigación de Operaciones, pero operan en horizontes de planificación (horas–días) más que en orquestación continua minuto a minuto.

**Simulación retrospectiva.** La DES ha sido ampliamente adoptada para evaluar políticas de despacho (SRPT, CR, ESI+FIFO) sin riesgo operativo (Applied Sciences, 2024). Revisiones comprehensivas confirman su valor metodológico, pero señalan que la mayoría de implementaciones utilizan datos históricos en reposo, sin telemetría en vivo ni capacidad de simular miles de escenarios estocásticos por minuto alimentados por el estado actual del departamento (Gunal & Roe, 2014; Klein et al., 2013).

**Deep Reinforcement Learning.** Liu et al. (2020) demostraron que agentes DRL pueden optimizar programación de pacientes en ED formulando el problema como MDP. Este precedente valida la idoneidad del DRL para decisiones secuenciales, pero no integra gemelo digital bidireccional ni capa XAI orientada a adopción clínica.

## 1.3 Brechas críticas de investigación

El análisis bibliográfico identifica tres fronteras inexploradas que constituyen oportunidades de contribución original:

**Brecha 1 — Simulación pasiva versus Gemelos Digitales activos y bidireccionales.** La DES retrospectiva no sincroniza con telemetría RTLS/EHR en tiempo real ni emite directrices operativas autónomas antes del colapso físico (Konios et al., NTU; automate.org, 2024).

**Brecha 2 — Orquestación mediante Deep Reinforcement Learning.** El aprendizaje supervisado predice sucesos; el DRL optimiza secuencias de decisiones. Existe escasez de investigación sobre agentes DQN que despachen dinámicamente recursos médicos balanceando demoras críticas, eficiencia global y equidad (Liu et al., 2020).

**Brecha 3 — XAI e Ingeniería de Factores Humanos.** Sistemas de caja negra enfrentan resistencia clínica en contextos de alta presión. Se requiere integración de SHAP/LIME con interfaces que traduzcan inferencias en recomendaciones operativas comprensibles (Vural et al., 2025; Adel et al., 2025).

## 1.4 Pregunta de investigación y objetivos

**Pregunta de investigación:** ¿De qué manera la integración de un Gemelo Digital activo-bidireccional, un agente DQN sobre MDP y una capa XAI transparente puede optimizar en tiempo real el despacho de recursos y el flujo de pacientes para mitigar el hacinamiento y reducir el LOS en servicios de urgencias?

**Objetivos específicos:**

1. Diseñar e implementar un Gemelo Digital SimPy con sincronización bidireccional (Δt ≤ 30 s) y proyección estocástica de escenarios.
2. Formular el ED como MDP y entrenar un agente DQN con función de recompensa multiobjetivo (demoras críticas, LOS, ocupación, throughput, equidad).
3. Desarrollar una capa XAI basada en SHAP con paneles diferenciados para clínicos y gestores operativos.
4. Validar computacionalmente el sistema frente a baselines clínicos (ESI+FIFO, SRPT, heurísticos) mediante métricas operativas con intervalos de confianza bootstrap.
