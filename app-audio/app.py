import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ASR_MODEL_ID = "openai/whisper-small"

SYSTEM_PROMPT = (
    "Eres un asistente cálido, empático y no invasivo para reflexión de diario "
    "y apoyo emocional inicial. Valida las emociones de la persona sin juzgar, "
    "ofrece reflexiones suaves o pasos pequeños y alcanzables, no realices "
    "diagnósticos médicos ni psicológicos, sugiere buscar apoyo profesional si "
    "percibes malestar intenso o persistente, y responde SIEMPRE en el mismo "
    "idioma en que escribió o habló el usuario."
)

IDIOMAS_AUDIO = {
    "Auto": None,
    "Español": "es",
    "English": "en",
    "Français": "fr",
    "Português": "pt",
}

EJEMPLOS = [
    [
        "Hoy tuve muchas entregas y me siento agotada, pero me cuesta desconectarme.",
        None,
        "Español",
    ],
    [
        "I feel overwhelmed with tasks today and could use a gentle reflection.",
        None,
        "English",
    ],
    [
        "Aujourd'hui j'ai ressenti beaucoup de fatigue et j'ai besoin de souffler.",
        None,
        "Français",
    ],
]

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


def transcribir_audio(audio_path, idioma_seleccionado="Auto"):
    if not audio_path:
        return ""
    kwargs = {"task": "transcribe"}
    codigo_idioma = IDIOMAS_AUDIO.get(idioma_seleccionado)
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
            max_new_tokens=220,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    nuevos_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(nuevos_tokens, skip_special_tokens=True).strip()


def responder_apoyo(texto, audio, idioma_audio="Auto"):
    texto_limpio = (texto or "").strip()
    texto_audio = transcribir_audio(audio, idioma_audio)

    if texto_limpio and texto_audio:
        entrada_combinada = (
            f"Texto escrito:\n{texto_limpio}\n\nAudio transcrito:\n{texto_audio}"
        )
    elif texto_limpio:
        entrada_combinada = texto_limpio
    elif texto_audio:
        entrada_combinada = texto_audio
    else:
        return "Escribe un mensaje o graba/sube un audio para comenzar a reflexionar."

    return generar_respuesta(entrada_combinada)


demo = gr.Interface(
    fn=responder_apoyo,
    title="Diario y Apoyo Emocional",
    description=(
        "Usa HuggingFaceTB/SmolLM2-1.7B-Instruct y transcripción de voz como "
        "apoyo reflexivo (sin sustituir atención profesional)."
    ),
    inputs=[
        gr.Textbox(lines=4, label="Escribe tu entrada de diario"),
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio"),
        gr.Dropdown(
            choices=["Auto", "Español", "English", "Français", "Português"],
            value="Auto",
            label="Idioma del audio",
        ),
    ],
    outputs=gr.Textbox(lines=8, label="Respuesta de apoyo"),
    examples=EJEMPLOS,
)

if __name__ == "__main__":
    demo.launch()