# 2. Marco Teórico y Conceptual

## 2.1 Simulación Discreta de Eventos y Gemelos Digitales Activos

### 2.1.1 Limitaciones de la DES retrospectiva

La Simulación de Eventos Discretos (DES) modela entidades (pacientes) que compiten por recursos (camas, consultorios, estaciones de triaje) mediante colas, tiempos de servicio y reglas de despacho. En modo retrospectivo, la DES calibra parámetros sobre logs históricos y evalúa contrafácticos offline. Esta aproximación falla ante dinámicas complejas porque: (a) no captura el estado operativo instantáneo del plano físico; (b) no permite retroalimentación en tiempo real cuando la simulación diverge de la realidad; (c) tiene latencia incompatible con decisiones que deben tomarse en minutos durante picos de demanda (Applied Sciences, 2024; Klein et al., 2013).

### 2.1.2 Definición formal de Gemelo Digital activo y bidireccional

Un **Gemelo Digital activo** se define como réplica virtual del ED que:

- **Sincroniza** con el plano físico mediante telemetría RTLS (posición de pacientes y personal) y EHR (acuity, vitals, órdenes clínicas) con latencia permisible Δt ≤ 30 s.
- **Proyecta** N escenarios estocásticos por ciclo de sincronización mediante mini-runs SimPy con estado inicial clonado del snapshot físico.
- **Retroalimenta** directrices operativas al plano físico cuando la divergencia entre ocupación observada y proyectada supera umbral ε.

Formalmente, sea **s_phys(t)** el estado físico en tiempo t y **s_virt(t)** el estado virtual. La sincronización bidireccional implementa:

- **Pull:** s_phys(t) → ingest telemetry → update twin state
- **Push:** run_scenarios(s_phys, N, horizon) → projected occupancy distribution
- **Feedback:** if |occ_phys − occ_virt| > ε then emit directive

La arquitectura técnica comprende: fuentes de datos (RTLS, EHR, sensores operacionales), motor SimPy, capa BidirectionalSync con `pull_from_physical()`, `run_lookahead()` y `feedback_directive()`.

### 2.1.3 Diferenciación respecto a simulaciones estáticas

| Dimensión | DES estática | Gemelo Digital activo |
|---|---|---|
| Datos | Históricos en reposo | Telemetría en vivo |
| Sincronización | Ninguna | Bidireccional, Δt ≤ 30 s |
| Escenarios | Uno por experimento | N >> 1 por minuto |
| Salida | Reporte post-hoc | Directriz operativa |

## 2.2 Deep Reinforcement Learning en Procesos de Decisión de Markov

### 2.2.1 Por qué DQN supera métodos predictivos aislados

El aprendizaje supervisado estima P(outcome | features) pero no optimiza **secuencias** de decisiones bajo incertidumbre. En ED, cada acción de despacho (priorizar ESI, activar fast-track, reasignar imagenología) altera el estado futuro del sistema. El Deep Reinforcement Learning modela esta interacción mediante un agente que maximiza recompensa acumulada descontada interactuando con el entorno (Liu et al., 2020).

### 2.2.2 Formalización del ED como MDP

Definimos el MDP **M = (S, A, P, R, γ)** donde:

- **S:** espacio de estados s ∈ R^17 codificando colas por ESI (5), tiempos de espera por ESI (5), ocupación de camas/consultorios/imagenología (3), hora del día (1), pacientes en sistema (1), SpO2 y FC promedio (2).
- **A:** conjunto discreto {0,…,5} de políticas de despacho compuestas (ESI estricto, refuerzo críticos, fast-track, prioridad imagenología, prioridad admisión, balance clínico-operativo).
- **P(s'|s,a):** transiciones estocásticas inducidas por llegadas NHPP, tiempos de servicio lognormales y política de despacho.
- **R(s,a,s'):** función de recompensa multiobjetivo.
- **γ ∈ [0,1):** factor de descuento (0.99 en implementación).

La función de recompensa instantánea:

**R_t = −α·(w_crit/60) − β·(LOS/120) − γ_occ·occ + δ·thr − λ_fair·(1 − equity)**

donde w_crit es espera media de pacientes ESI 1–2, LOS es length of stay, occ es ocupación normalizada, thr es throughput por hora, equity es índice de equidad inverso al Gini de esperas por ESI.

**Función de valor:** V^π(s) = E[Σ γ^t R_t | s_0=s, π]

**Función Q:** Q^π(s,a) = E[Σ γ^t R_t | s_0=s, a_0=a, π]

### 2.2.3 Arquitectura DQN

El Deep Q-Network aproxima Q(s,a;θ) con red neuronal feedforward:

- **Entrada:** vector s ∈ R^17
- **Capas ocultas:** [256, 256, 128] con activación ReLU
- **Salida:** Q-values para cada a ∈ A (dimensión 6)
- **Entrenamiento:** experience replay, target network (actualización cada 100 pasos), ε-greedy (ε: 1.0 → 0.05)
- **Optimizador:** Adam, lr = 10^−4, batch = 64, buffer = 50.000 transiciones
- **Loss:** Huber (smooth L1) entre Q(s,a;θ) y target r + γ max_a' Q(s',a';θ−)

## 2.3 Explainable AI e Ingeniería de Factores Humanos

### 2.3.1 Barreras de adopción clínica

En contextos críticos, la automatización sin transparencia genera resistencia justificada: clínicos deben poder auditar, cuestionar y anular recomendaciones algorítmicas sin comprometer responsabilidad profesional (Adel et al., 2025). La XAI no es ornamento de interfaz sino requisito de confianza operativa.

### 2.3.2 Integración SHAP sobre Q-values

Aplicamos SHAP (SHapley Additive exPlanations) para descomponer la decisión del agente en atribuciones por covariable operativa y fisiológica. KernelExplainer sobre la red Q(s,·) identifica las cinco features con mayor |φ_i| para la acción seleccionada.

### 2.3.3 Traducción a recomendaciones operativas

La capa XAI genera dos narrativas:

- **Panel clínico:** "Recomendación: Refuerzo casos críticos — debido a espera ESI-2 elevada (22 min) y SpO2 promedio 96.5%"
- **Panel gestor:** KPIs determinantes con pesos SHAP y nota de apoyo a la decisión (requiere validación del equipo de turno)

Vural et al. (2025) establecen precedente de frameworks explicables para predicción de hacinamiento; este trabajo extiende la explicabilidad a decisiones de orquestación DRL.
