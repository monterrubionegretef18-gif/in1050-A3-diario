# Investigación de la Problemática: Asistente Emocional IA



## 1. El problema

Los estudiantes universitarios enfrentan altos niveles de estrés académico, sobrecarga y ansiedad, lo que incrementa el riesgo de crisis emocionales. Con frecuencia, carecen de acceso inmediato a herramientas de autorreflexión que les permitan procesar sus emociones a tiempo o identificar patrones negativos antes de llegar al desgaste total.



* **Dato 1:** De acuerdo con la Organización Mundial de la Salud (OMS), una de cada cuatro personas sufre trastornos de salud mental en algún momento de su vida, siendo la etapa universitaria un periodo crítico de vulnerabilidad al estrés crónico \[Fuente: OMS - Salud Mental en el Entorno Universitario].

* **Dato 2:** Estudios recientes sobre bienestar estudiantil indican que más del 60% de los universitarios reportan síntomas moderados o severos de ansiedad vinculados a la presión académica, y un porcentaje elevado no busca apoyo profesional a tiempo por falta de orientación inicial \[Fuente: INEGI / Reportes de Bienestar Estudiantil en México].



## 2. Cómo se resuelve hoy

Actualmente, las alternativas para el manejo del estrés y la salud emocional de los estudiantes presentan limitaciones en accesibilidad y inmediatez:



| Solución / Alternativa | Qué ofrece | Limitaciones |
|---|---|---|
| **Terapias y líneas de ayuda institucionales** | Atención psicológica profesional y personalizada. | Tiempos de espera prolongados para citas y horarios limitados fuera de guardias universitarias. |
| **Aplicaciones de meditación (Headspace, Calm)** | Guías generales de relajación y respiración. | Contenido genérico que no analiza la situación académica específica ni detecta distorsiones cognitivas particulares. |
| **Diarios personales tradicionales** | Espacio libre para desahogo sin restricciones de formato. | Falta de retroalimentación analítica, orientación estructurada o detección de señales de alerta de crisis. |



## 3. Por qué con IA y por qué este modelo

El uso de modelos de lenguaje ligeros permite procesar fragmentos de diario personal de forma local y privada, brindando un análisis cualitativo inmediato del estado de ánimo y detectando patrones de pensamiento distorsionados de manera automatizada.



### Comparativa de Modelos en Hugging Face



| Criterio | Modelo Seleccionado: SmolLM2-1.7B-Instruct | Modelo Alternativo: Llama-3-8B-Instruct |
|---|---|---|
| **Parámetros** | 1.7 Mil Millones (1.7B) | 8 Mil Millones (8B) |
| **Licencia** | Apache 2.0 | Meta Llama 3 Community License |
| **Ventajas** | Alta eficiencia en recursos, requiere baja VRAM (\~3.4 GB en FP16) y se ejecuta de forma rápida y gratuita en Google Colab. | Mayor capacidad contextual global. |
| **Desventajas** | Menor profundidad en razonamientos extremadamente complejos en comparación con modelos más grandes. | Consumo elevado de recursos de hardware y requerimientos estrictos de infraestructura para ejecución fluida. |



## 4. Usuarios y requisitos

### Usuarios objetivo

Estudiantes universitarios que experimentan altos niveles de estrés, ansiedad o sobrecarga académica y buscan un espacio seguro de autorreflexión.


### Requisitos del Sistema

* **Requisito 1:** Interfaz de entrada para fragmentos de diario personal o textos breves sobre el estado emocional diario.

* **Requisito 2:** Evaluación cualitativa del estado de ánimo mediante el modelo `SmolLM2-1.7B-Instruct`.

* **Requisito 3:** Identificación de patrones de distorsión cognitiva en el discurso del estudiante.

* **Requisito 4:** Mecanismo de canalización profesional con recomendaciones y líneas de ayuda automáticas ante situaciones de riesgo.



## 5. Impacto ético

* **Privacidad de los datos:** Los diarios personales contienen información altamente sensible. Se prioriza la ejecución local o controlada en entornos seguros para proteger la confidencialidad del usuario.

* **Sesgos y limitaciones:** El modelo puede generar respuestas sesgadas por datos web o malinterpretar la severidad de una crisis. Se declara explícitamente que **no sustituye el asesoramiento médico o terapéutico profesional**.

* **Guardrails de seguridad (Crítico):** Implementación de reglas estrictas por palabras clave (Hardcoded Guardrails) para la detección inmediata de señales de alerta severas (como ideación suicida o conductas de riesgo), interrumpiendo la respuesta común del modelo para desplegar de forma directa números de emergencia profesional.

* **Costo energético:** La optimización de un modelo de 1.7B parámetros reduce de manera significativa la huella computacional y energética en comparación con arquitecturas masivas.



## 6. Referencias

1. [OMS - Salud Mental: Fortalecer nuestra respuesta](https://www.who.int/es/news-room/fact-sheets/detail/mental-health-strengthening-our-response)

2. [Hugging Face - SmolLM2-1.7B-Instruct Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

3. [Documentación de la librería Transformers de Hugging Face](https://huggingface.co/docs/transformers/index)

4. [Lineamientos Éticos para la Inteligencia Artificial en Aplicaciones de Salud Mental](https://huggingface.co/)

   [Guía de Recursos de Atención Psicológica Universitaria - Ibero León](https://www.iberoleon.mx/)

