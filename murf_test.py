from murf import Murf

client = Murf(api_key="ap2_909d4dac-d4aa-477c-a2a5-cb7d02af060d")

response = client.text_to_speech.generate(
    text="This is Priya testing Murf's voice API!",
    voice_id="es-ES-carla",   # ✅ Try other voices too (can help you list them if needed)
    style="Conversational",   # Optional style
    multi_native_locale="en-IN"  # Optional locale
)

print("Audio URL:", response.audio_file)
