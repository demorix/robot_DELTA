import numpy as np
import requests
import sounddevice as sd

class TTS:
    def __init__(self):
        """Ініціалізація налаштувань для ElevenLabs"""
        self.api_key = "sk_40b3750b3cd80a3209387f256cc449f14724f42c0bd42d01"  
        self.voice_id = "ErXwobaYiN019PkySvjV"
        
        self.volume = 1.0 

    def synthesize(self, text):
        """Отримує сирі байти (24000 Гц) з хмари і миттєво їх відтворює"""
        if not text.strip() or self.api_key == "ТВІЙ_КЛЮЧ_ТУТ":
            return

        # Використовуємо дозволений безкоштовний формат pcm_24000
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}?output_format=pcm_24000"
        
        headers = {
            "Accept": "audio/pcm", 
            "Content-Type": "application/json", 
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text, 
            "model_id": "eleven_multilingual_v2", 
            "voice_settings": {
                "stability": 0.5, 
                "similarity_boost": 0.75
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # 1. Отримуємо сирі байти
                audio_bytes = response.content
                
                # 2. Перетворюємо у числа
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0
                
                # 3. Застосовуємо гучність
                audio_float = audio_float * self.volume
                
                # 4. Відтворюємо звук на частоті 24000
                sd.play(audio_float, samplerate=24000)
                sd.wait()
            else:
                pass
                
        except Exception:
            pass
