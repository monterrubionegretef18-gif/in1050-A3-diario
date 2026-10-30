# Diario Inteligente: Herramienta de Autorreflexión y Apoyo Emocional

Un sistema interactivo de análisis emocional para jóvenes universitarios con altos niveles de estrés. El usuario escribe fragmentos de su diario personal y la herramienta procesa el texto para generar análisis cualitativos de su estado de ánimo, identificar distorsiones cognitivas comunes y ofrecer recomendaciones personalizadas de apoyo emocional y canalización profesional.

## Equipo

| Nombre | Papel en el proyecto |
| :--- | :--- |
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

## Infraestructura

El cálculo de memoria mínima estimada para cargar el modelo en precisión FP16 es el siguiente:

$$ \text{Memoria} = 1.7 \times 10^9 \text{ parámetros} \times 2 \text{ bytes} \approx 3.4 \text{ GB} $$

**Decisión de Infraestructura:** El modelo se ejecutará en **Google Colab con GPU T4** (15 GB de VRAM de entorno gratuito). Esta plataforma cubre holgadamente los ~3.4 GB requeridos para los pesos y la memoria del KV-cache durante la inferencia, permitiendo un procesamiento local rápido, sin costo y sin saturar memoria de procesamiento.

## Riesgo Ético

* **Riesgo Principal:** Que el modelo genere un diagnóstico erróneo, responda de forma inadecuada o minimice una crisis severa de salud mental (como ideación suicida).
* **Mitigación:** Implementación de un filtro estricto por reglas de palabras clave (*hardcoded guardrail*) que detecte indicadores de peligro e interrumpa al modelo para mostrar de inmediato números telefónicos de emergencia profesional, acompañado de un aviso explícito aclarando que la aplicación no sustituye a un terapeuta humano.

## Cambios respecto a la Fase 1

* **Aclaración del idioma:** Se integró en la lógica de indicación (*prompting*) una instrucción explícita para asegurar respuestas fluidas y empáticas en español, mitigando la limitación del modelo referente al idioma inglés.
* **Reforzamiento del Guardrail:** Se definieron listas concretas de palabras clave de alerta prioritaria antes de invocar la inferencia del modelo.

## Siguientes Pasos

- [ ] Crear la interfaz interactiva para el usuario final (usando Gradio / Streamlit en Colab).
- [ ] Optimizar las plantillas de prompts para afinar el tono empático y la precisión en la identificación de distorsiones cognitivas.
- [ ] Realizar pruebas de estrés con casos de uso reales y validar la correcta activación del guardrail de seguridad.
- [ ] Diseñar y presentar el informe final y prototipo ejecutable para el Parcial III.

## Referencias

1. Hugging Face. (2024). *SmolLM2-1.7B-Instruct Model Card*. [https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
2. Allsopp, K., et al. (2019). *Heterogeneity in psychiatric diagnostic classification*. Psychiatry Research. [https://doi.org/10.1016/j.psychres.2019.07.005](https://doi.org/10.1016/j.psychres.2019.07.005)

