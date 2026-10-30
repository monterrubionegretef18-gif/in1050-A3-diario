# Diario Inteligente: Herramienta de Autorreflexión y Apoyo Emocional

Un sistema interactivo de análisis emocional para jóvenes universitarios con altos niveles de estrés. El usuario escribe fragmentos de su diario personal y la herramienta procesa el texto para generar análisis cualitativos de su estado de ánimo, identificar distorsiones cognitivas comunes y ofrecer recomendaciones personalizadas de apoyo emocional y canalización profesional.

## Equipo

| Nombre | Papel en el proyecto |
| :--- | :--- |
| Ian Alejandro Maciel Romo | Desarrollo e Integración del Modelo |
| José Arturo Amaro Casillas | Investigación de Problemática y Documentación |
| Fernanda Montes Negrete | Diseño de Interfaz y Lógica de Prompting |
| Paola Romina Meza Lopez | Análisis de Riesgos Éticos y Guardrails |

## Problemática

La falta de acceso inmediato a herramientas de autorreflexión y apoyo emocional temprano para jóvenes universitarios propicia un incremento en el estrés, ansiedad y riesgo de crisis emocionales no atendidas a tiempo. Para conocer la investigación detallada de esta problemática, consulta nuestro documento de [Investigación de la Problemática](docs/investigacion.md).

## Caso de Uso

* **Usuario:** Estudiante universitario con altos niveles de estrés o ansiedad.
* **Entrada del sistema:** Fragmentos de texto del diario personal o descripción diaria del estado de ánimo expresados por el estudiante.
* **Salida del sistema:** Un análisis cualitativo del estado emocional, identificación de posibles distorsiones cognitivas (como la sobregeneralización) y recomendaciones de afrontamiento junto con recursos de ayuda.

## Modelo

| Identificador | Tarea | Núm. Parámetros | Licencia | Limitación Declarada |
| :--- | :--- | :--- | :--- | :--- |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | `text-generation` | 1.71 mil millones (1.7B) | Apache 2.0 | Los modelos entienden y generan principalmente contenido en inglés; el contenido generado puede no ser siempre factualmente preciso o libre de sesgos, por lo que no debe usarse como sustituto de asesoramiento médico o profesional. |

## Cómo Funciona

```mermaid
graph LR
    A[Entrada: Texto del Diario] --> B[Filtro Guardrail de Seguridad]
    B -- Detección de Crisis Severa --> C[(Módulo de Emergencia: Líneas de Ayuda)]
    B -- Estado Normal / Estrés --> D[SmolLM2-1.7B-Instruct]
    D --> E[Salida: Análisis Emocional y Recomendaciones]
cat << 'EOF' > README.md
# Diario Inteligente: Herramienta de Autorreflexión y Apoyo Emocional

Un sistema interactivo de análisis emocional para jóvenes universitarios con altos niveles de estrés. El usuario escribe fragmentos de su diario personal y la herramienta procesa el texto para generar análisis cualitativos de su estado de ánimo, identificar distorsiones cognitivas comunes y ofrecer recomendaciones personalizadas de apoyo emocional y canalización profesional.

## Equipo

| Nombre | Papel en el proyecto |
| :--- | :--- |
| Ian Alejandro Maciel Romo | Desarrollo e Integración del Modelo |
| José Arturo Amaro Casillas | Investigación de Problemática y Documentación |
| Fernanda Montes Negrete | Diseño de Interfaz y Lógica de Prompting |
| Paola Romina Meza Lopez | Análisis de Riesgos Éticos y Guardrails |

## Problemática

La falta de acceso inmediato a herramientas de autorreflexión y apoyo emocional temprano para jóvenes universitarios propicia un incremento en el estrés, ansiedad y riesgo de crisis emocionales no atendidas a tiempo. Para conocer la investigación detallada de esta problemática, consulta nuestro documento de [Investigación de la Problemática](docs/investigacion.md).

## Caso de Uso

* **Usuario:** Estudiante universitario con altos niveles de estrés o ansiedad.
* **Entrada del sistema:** Fragmentos de texto del diario personal o descripción diaria del estado de ánimo expresados por el estudiante.
* **Salida del sistema:** Un análisis cualitativo del estado emocional, identificación de posibles distorsiones cognitivas (como la sobregeneralización) y recomendaciones de afrontamiento junto con recursos de ayuda.

## Modelo

| Identificador | Tarea | Núm. Parámetros | Licencia | Limitación Declarada |
| :--- | :--- | :--- | :--- | :--- |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | `text-generation` | 1.71 mil millones (1.7B) | Apache 2.0 | Los modelos entienden y generan principalmente contenido en inglés; el contenido generado puede no ser siempre factualmente preciso o libre de sesgos, por lo que no debe usarse como sustituto de asesoramiento médico o profesional. |

## Cómo Funciona

```mermaid
flowchart TD
A([Usuario])-->B[Entrada deL usuario]-->C((SmolLM2-1.7B-Instruct))-->D[Modelo hace su transcripción]-->E{¿Es una situación de riesgo?}
E--> |Si| F[Conexión con un profesional]
E--> |No| G[Recomendar al usuario]

```

