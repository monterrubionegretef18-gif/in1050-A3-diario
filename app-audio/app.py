import time
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ASR_MODEL_ID = "openai/whisper-small"

SYSTEM_PROMPT = (
    "Eres un consejero empático de apoyo emocional para estudiantes universitarios. "
    "Responde SIEMPRE en español natural, claro, breve (máximo 3 o 4 oraciones) y sin errores de traducción.\n"
    "Tu objetivo en cada respuesta es:\n"
    "1. Validar con calidez cómo se siente el estudiante.\n"
    "2. Ayudarle a cuestionar pensamientos exagerados (como creer que todo saldrá mal siempre).\n"
    "3. Darle un consejo práctico y hacerle una pregunta corta para seguir platicando.\n\n"
    "Ejemplo 1:\n"
    "Usuario: Me siento mal porque reprobé una materia en la uni.\n"
    "Asistente: Lamento mucho que estés pasando por esto, reprobar una materia es muy frustrante pero no define tu capacidad como estudiante. "
    "A veces sentimos que un tropiezo arruina toda la carrera, pero aún estás a tiempo de recuperarte. "
    "Te recomiendo revisar en qué temas fallaste y hablar con tu profesor; ¿qué materia fue y cómo te sientes ahora?\n\n"
    "Ejemplo 2:\n"
    "Usuario: Llevo días muy estresado y siento que no voy a poder con todo.\n"
    "Asistente: Es completamente normal sentirte abrumado cuando se juntan muchas entregas. "
    "En lugar de pensar en todo a la vez, intenta elegir solo una tarea pequeña para terminar hoy y date un respiro. "
    "Si este agobio lleva mucho tiempo, recuerda que puedes acudir al área de apoyo psicológico de tu universidad. ¿Qué es lo que más te preocupa ahorita?"
)

# Nivel 3: Hardcoded Guardrail para crisis severa (interrumpe el modelo)
PALABRAS_CRISIS = [
    "suicidio",
    "suicidarme",
    "matarme",
    "quitarme la vida",
    "no quiero vivir",
    "ya no quiero vivir",
    "hacerme daño",
    "cortarme",
    "morirme",
    "suicide",
    "kill myself",
    "end my life",
    "don't want to live",
]

# Nivel 2: Palabras de temas problemáticos que activan refuerzo de recomendaciones profesionales
PALABRAS_DELICADAS = [
    "depresivo",
    "depresión",
    "deprimido",
    "ataque de pánico",
    "ansiedad severa",
    "violencia",
    "abuso",
    "adicción",
    "no puedo más",
    "desesperado",
    "desesperada",
]

MENSAJE_EMERGENCIA = (
    "🚨 **Me preocupa mucho lo que me cuentas y quiero que estés a salvo.** 🚨\n\n"
    "No tienes que cargar con esto solo. Por favor, busca apoyo profesional en este momento:\n"
    "• **Línea de la Vida (México, 24/7 gratuita):** 800 911 2000\n"
    "• **Emergencias Nacionales:** 911\n"
    "• **SAPTEL (Apoyo psicológico gratuito):** 55 5259 8121\n"
    "• **Apoyo Universitario:** Acércate hoy mismo al área de atención psicológica de tu campus o con alguien de tu confianza.\n\n"
    "*Este espacio es una herramienta automatizada de apoyo y no sustituye la atención de un profesional.*"
)

MAPA_IDIOMAS = {
    "Auto": None,
    "Español": "spanish",
    "English": "english",
    "Français": "french",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)
model.to(device)
model.eval()

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=ASR_MODEL_ID,
    device=0 if device == "cuda" else -1,
)


def transcribir_audio(audio_path, idioma_seleccionado):
    if not audio_path:
        return ""
    kwargs = {"task": "transcribe"}
    codigo_idioma = MAPA_IDIOMAS.get(idioma_seleccionado)
    if codigo_idioma:
        kwargs["language"] = codigo_idioma

    resultado = asr_pipe(audio_path, generate_kwargs=kwargs)
    if isinstance(resultado, dict):
        return resultado.get("text", "").strip()
    return str(resultado).strip()


def generar_respuesta_conversacional(historial_mensajes):
    mensajes_modelo = [{"role": "system", "content": SYSTEM_PROMPT}] + historial_mensajes[-6:]

    prompt = tokenizer.apply_chat_template(
        mensajes_modelo, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=140,
            temperature=0.25,
            top_p=0.85,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    nuevos_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(nuevos_tokens, skip_special_tokens=True).strip()


def conversar(texto, audio, idioma, historial_chat, historial_llm):
    inicio = time.time()

    texto_limpio = (texto or "").strip()
    texto_audio = transcribir_audio(audio, idioma)

    if texto_limpio and texto_audio:
        entrada_usuario = f"{texto_limpio} (Audio: {texto_audio})"
    elif texto_limpio:
        entrada_usuario = texto_limpio
    elif texto_audio:
        entrada_usuario = texto_audio
    else:
        return (
            historial_chat,
            historial_llm,
            "",
            "0.00 s",
            "",
            None,
        )

    texto_lower = entrada_usuario.lower()

    # Nivel 3: Hardcoded Guardrail de crisis inmediata
    if any(palabra in texto_lower for palabra in PALABRAS_CRISIS):
        respuesta = MENSAJE_EMERGENCIA
    else:
        mensaje_para_llm = entrada_usuario
        if any(palabra in texto_lower for palabra in PALABRAS_DELICADAS):
            mensaje_para_llm += (
                " (Responde con mucha preocupación, dale 2 recomendaciones claras de qué hacer "
                "y sugiérele acudir con un psicólogo o apoyo universitario, cerrando con una pregunta breve)."
            )

        historial_llm.append({"role": "user", "content": mensaje_para_llm})
        respuesta = generar_respuesta_conversacional(historial_llm)
        historial_llm.append({"role": "assistant", "content": respuesta})

    historial_chat.append({"role": "user", "content": entrada_usuario})
    historial_chat.append({"role": "assistant", "content": respuesta})

    tiempo_total = f"{time.time() - inicio:.2f} s"
    return historial_chat, historial_llm, entrada_usuario, tiempo_total, "", None


def limpiar_conversacion():
    return [], [], "", "0.00 s", "", None


with gr.Blocks(title="Diario y Apoyo Emocional — Equipo 3") as demo:
    gr.Markdown(
        "# Diario y Apoyo Emocional — Equipo 3 (Ibero León)\n"
        "Asistente conversacional impulsado por **HuggingFaceTB/SmolLM2-1.7B-Instruct** y **openai/whisper-small** "
        "para autorreflexión emocional continua.\n\n"
        "⚠️ **Aviso ético:** Este sistema es únicamente una herramienta de apoyo reflexivo; "
        "**no** realiza diagnósticos médicos ni sustituye la atención de un profesional de la salud."
    )

    estado_llm = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            entrada_texto = gr.Textbox(
                lines=3,
                placeholder="Cuéntame cómo te sientes hoy o responde para seguir platicando...",
                label="Escribe tu entrada de diario o mensaje",
            )
            entrada_audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="O graba / sube un audio",
            )
            selector_idioma = gr.Dropdown(
                choices=["Auto", "Español", "English", "Français"],
                value="Auto",
                label="Idioma del audio",
            )
            with gr.Row():
                btn_limpiar = gr.Button("Nueva conversación", variant="secondary")
                btn_enviar = gr.Button("Enviar", variant="primary")

            salida_transcripcion = gr.Textbox(
                lines=2, label="Última entrada procesada / Transcripción"
            )
            salida_tiempo = gr.Textbox(label="Tiempo de respuesta")

        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                height=480,
                label="Conversación de apoyo y autorreflexión",
            )

    gr.Examples(
        examples=[
            [
                "Me siento muy mal porque reprobé una materia en la uni y siento que no sirvo para esto.",
                "Español",
            ],
            [
                "Llevo semanas sintiéndome muy depresivo y sin ganas de hacer nada, ¿qué puedo hacer?",
                "Español",
            ],
            [
                "I feel overwhelmed with tasks today and could use a gentle reflection.",
                "English",
            ],
            [
                "Ya no aguanto la presión, siento que no quiero vivir más.",
                "Español",
            ],
        ],
        inputs=[entrada_texto, selector_idioma],
    )

    btn_enviar.click(
        fn=conversar,
        inputs=[entrada_texto, entrada_audio, selector_idioma, chatbot, estado_llm],
        outputs=[
            chatbot,
            estado_llm,
            salida_transcripcion,
            salida_tiempo,
            entrada_texto,
            entrada_audio,
        ],
    )

    entrada_texto.submit(
        fn=conversar,
        inputs=[entrada_texto, entrada_audio, selector_idioma, chatbot, estado_llm],
        outputs=[
            chatbot,
            estado_llm,
            salida_transcripcion,
            salida_tiempo,
            entrada_texto,
            entrada_audio,
        ],
    )

    btn_limpiar.click(
        fn=limpiar_conversacion,
        outputs=[
            chatbot,
            estado_llm,
            salida_transcripcion,
            salida_tiempo,
            entrada_texto,
            entrada_audio,
        ],
    )

if __name__ == "__main__":
    demo.launch()