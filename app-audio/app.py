import time
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ASR_MODEL_ID = "openai/whisper-small"

SYSTEM_PROMPT = (
    "Eres un asistente empático de autorreflexión emocional para estudiantes universitarios. "
    "Responde SIEMPRE en el MISMO IDIOMA en que escribió o habló el usuario (español, inglés o francés), "
    "con redacción natural, cálida y sin errores gramaticales.\n"
    "Organiza tu respuesta exactamente en estas 3 secciones breves:\n"
    "1. Validación emocional: Reconoce cómo se siente la persona sin juzgar ni dar falsas esperanzas.\n"
    "2. Distorsión cognitiva detectada: Identifica si hay algún patrón de pensamiento negativo "
    "(ej. catastrofismo, sobregeneralización, pensamiento todo o nada, lectura de mente) y explícalo en una frase.\n"
    "3. Paso práctico para hoy: Sugiere una acción pequeña, realista y saludable de autocuidado.\n"
    "No des diagnósticos médicos y recuerda buscar apoyo profesional si el malestar es intenso."
)

# Hardcoded Guardrail: Palabras clave de crisis severa (Fase 1)
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

MENSAJE_EMERGENCIA = (
    "🚨 ALERTA DE APOYO INMEDIATO 🚨\n\n"
    "Detectamos que podrías estar pasando por un momento muy difícil o de riesgo. "
    "No estás solo y tu bienestar es lo más importante. Por favor, comunícate de inmediato "
    "con un profesional o servicio de emergencia:\n\n"
    "• Línea de la Vida (México, 24/7 gratuita): 800 911 2000\n"
    "• Emergencias Nacionales: 911\n"
    "• SAPTEL (Apoyo psicológico gratuito): 55 5259 8121\n"
    "• Atención psicológica universitaria: Acude al área de apoyo psicopedagógico de tu campus.\n\n"
    "Nota: Este sistema es una herramienta automatizada y no sustituye la atención de un profesional de la salud."
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


def generar_respuesta(mensaje_usuario):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": mensaje_usuario},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=230,
            temperature=0.45,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    nuevos_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(nuevos_tokens, skip_special_tokens=True).strip()


def responder_apoyo(texto, audio, idioma):
    inicio = time.time()

    texto_limpio = (texto or "").strip()
    texto_audio = transcribir_audio(audio, idioma)

    if texto_limpio and texto_audio:
        entrada_combinada = (
            f"Texto escrito: {texto_limpio}\nAudio transcrito: {texto_audio}"
        )
    elif texto_limpio:
        entrada_combinada = texto_limpio
    elif texto_audio:
        entrada_combinada = texto_audio
    else:
        tiempo_total = f"{time.time() - inicio:.2f} s"
        return (
            "",
            "Escribe un mensaje o graba/sube un audio para comenzar a reflexionar.",
            tiempo_total,
        )

    # Hardcoded Guardrail: interrumpe el modelo si detecta palabras de crisis
    if any(palabra in entrada_combinada.lower() for palabra in PALABRAS_CRISIS):
        tiempo_total = f"{time.time() - inicio:.2f} s"
        return entrada_combinada, MENSAJE_EMERGENCIA, tiempo_total

    respuesta = generar_respuesta(entrada_combinada)
    tiempo_total = f"{time.time() - inicio:.2f} s"
    return entrada_combinada, respuesta, tiempo_total


ejemplos = [
    [
        "Siento que por más que estudio y me desvelo para los exámenes, no importa lo que haga, siempre voy a reprobar todo. Ya ni vale la pena seguir intentándolo.",
        None,
        "Español",
    ],
    [
        "I feel overwhelmed with tasks today and could use a gentle reflection to organize my mind.",
        None,
        "English",
    ],
    [
        "Aujourd'hui j'ai ressenti beaucoup de fatigue et j'ai besoin d'un moment de calme.",
        None,
        "Français",
    ],
    [
        "Ya no aguanto la presión, siento que no quiero vivir más.",
        None,
        "Español",
    ],
]

demo = gr.Interface(
    fn=responder_apoyo,
    inputs=[
        gr.Textbox(
            lines=4,
            placeholder="Escribe aquí cómo te sientes hoy...",
            label="Escribe tu entrada de diario",
        ),
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio"),
        gr.Dropdown(
            choices=["Auto", "Español", "English", "Français"],
            value="Auto",
            allow_custom_value=False,
            label="Idioma del audio",
        ),
    ],
    outputs=[
        gr.Textbox(lines=2, label="Entrada procesada / Transcripción"),
        gr.Textbox(lines=9, label="Análisis emocional y distorsiones cognitivas"),
        gr.Textbox(label="Tiempo de respuesta"),
    ],
    title="Diario y Apoyo Emocional — Equipo 3 (Ibero León)",
    description=(
        "Herramienta impulsada por **HuggingFaceTB/SmolLM2-1.7B-Instruct** y **openai/whisper-small** "
        "para autorreflexión emocional e identificación de distorsiones cognitivas.\n\n"
        "⚠️ **Aviso ético:** Este sistema es únicamente una herramienta de apoyo reflexivo; "
        "**no** realiza diagnósticos médicos ni sustituye la atención de un profesional de la salud."
    ),
    examples=ejemplos,
)

if __name__ == "__main__":
    demo.launch()