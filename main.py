import nemo
import nemo.collections.asr as nemo_asr

# Cargar modelo preentrenado
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Transcribir audio en español
audio_file = "audio.wav"
transcription = asr_model.transcribe([audio_file])
print(transcription)
