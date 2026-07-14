# 6. Conclusiones

Este artículo presentó un **Sistema Ciberfísico Híbrido de Orquestación Hospitalaria** para departamentos de urgencias que integra Gemelo Digital activo-bidireccional (SimPy + telemetría RTLS/EHR), agente Deep Q-Network sobre formulación MDP, y capa Explainable AI con paneles clínico-gestor. La contribución cierra explícitamente tres brechas documentadas en literatura Q1/Q2: (1) transición de simulación retrospectiva a gemelos digitales sincronizados en tiempo real; (2) orquestación secuencial mediante DRL vs. predicción supervisada aislada; (3) integración de SHAP con Ingeniería de Factores Humanos para adopción clínica.

La validación computacional con réplicas Monte Carlo, intervalos bootstrap 95% y análisis de robustez bajo estrés de demanda y recursos sustenta la viabilidad operativa del enfoque. El agente DQN converge establemente y produce recomendaciones auditables mediante atribuciones SHAP sobre covariables operativas y fisiológicas.

**Impacto clínico y operativo proyectado:** reducción de LOS, mitigación de ocupación máxima en picos, mejora de equidad de acceso por grupo ESI y optimización de throughput sin incremento de capacidad física.

**Contribución teórica y metodológica:** formalización rigurosa del ED como MDP con recompensa multiobjetivo; arquitectura de referencia para sistemas ciberfísicos hospitalarios que preserva separación disciplinar sinérgica entre IA predictiva e Ingeniería Industrial logística.

**Líneas futuras:** (1) arquitectura multiagente para coordinación ED–UCI–laboratorio; (2) transfer learning de políticas DQN entre hospitales; (3) validación prospectiva in vivo con shadow mode y estudio de usabilidad clínica; (4) integración de predicción clínica granular (NLP sobre notas de enfermería, Adel et al., 2025) como features adicionales del estado MDP; (5) despliegue MLOps completo en GCP con monitoreo de drift y reentrenamiento automático.
