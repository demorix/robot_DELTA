import os
import time
import wave
import torch
import numpy as np
import requests
import pygame

# === ІНІЦІАЛІЗАЦІЯ АУДІО ===
pygame.mixer.init()

# === НАЛАШТУВАННЯ ===
ELEVENLABS_API_KEY = "sk_40b3750b3cd80a3209387f256cc449f14724f42c0bd42d01"  # Вставте свій токен!
ELEVENLABS_VOICE_ID = "ErXwobaYiN019PkySvjV" 
ELEVENLABS_OUTPUT = "11labs_output.mp3"

SILERO_OUTPUT = "silero_output.wav"
local_model_path = 'silero_ua.pt'

# Змінна для зберігання моделі (завантажимо пізніше, якщо буде потрібно)
silero_model = None 

def say(text, model_name, volume=1.0):
    # Звільняємо аудіоплеєр (захист Windows)
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    try:
        pygame.mixer.music.unload()
    except AttributeError:
        pass

    play_file = None

    # --- ЛОГІКА ELEVENLABS ---
    if model_name == "elevenlabs":
        if ELEVENLABS_API_KEY == "ТВІЙ_КЛЮЧ_ТУТ":
            print("Помилка: Ви забули вставити API ключ ElevenLabs!")
            return
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg", 
            "Content-Type": "application/json", 
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": text, 
            "model_id": "eleven_multilingual_v2", 
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                with open(ELEVENLABS_OUTPUT, 'wb') as f:
                    f.write(response.content)
                play_file = ELEVENLABS_OUTPUT
            else:
                print(f"Помилка 11labs: {response.status_code} - {response.text}")
                return
        except Exception as e:
            print(f"Помилка з'єднання з хмарою: {e}")
            return

    # --- ЛОГІКА SILERO ---
    elif model_name == "silero":
        try:
            audio_tensor = silero_model.apply_tts(text=text, speaker='mykyta', sample_rate=48000)
            audio_np = audio_tensor.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            with wave.open(SILERO_OUTPUT, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(audio_int16.tobytes())
            
            play_file = SILERO_OUTPUT
        except Exception as e:
            print(f"Помилка Silero: {e}")
            return

    # --- ВІДТВОРЕННЯ ---
    if play_file:
        pygame.mixer.music.load(play_file)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()

if __name__ == "__main__":
    print("\n" + "="*40)
    print(" ГОЛОСОВИЙ МОДУЛЬ РОБОТА ")
    print("="*40)
    print("Виберіть модель для поточного сеансу:")
    print("1 - Silero (Локально, безкоштовно, для відлагодження)")
    print("2 - ElevenLabs (Хмарно, преміум-якість, для демонстрації)")
    
    # Цикл вибору моделі
    while True:
        choice = input("\nВаш вибір (введіть 1 або 2): ").strip()
        
        if choice == '1':
            active_model = "silero"
            print("\nЗавантаження локальної моделі Silero... зачекайте.")
            if not os.path.isfile(local_model_path):
                torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ua/v3_ua.pt', local_model_path)
            silero_model = torch.package.PackageImporter(local_model_path).load_pickle("tts_models", "model")
            silero_model.to(torch.device('cpu'))
            print("Silero готова!")
            break
            
        elif choice == '2':
            active_model = "elevenlabs"
            print("\nВибрано ElevenLabs. Перевірка наявності ключа...")
            if ELEVENLABS_API_KEY == "ТВІЙ_КЛЮЧ_ТУТ":
                print("УВАГА: Ви не вставили API ключ у код! Модуль не працюватиме.")
            else:
                print("ElevenLabs готова!")
            break
            
        else:
            print("Неправильний вибір. Будь ласка, введіть цифру 1 або 2.")

    # Основний цикл спілкування
    print(f"\nРобот увімкнений. Поточна модель: {active_model.upper()}")
    say("Система готова до роботи.", model_name=active_model)
    
    while True:
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        txt = input("\nВведіть текст (або 'q' для виходу): ")
        
        if txt.lower() == 'q': 
            break
            
        if txt.strip():
            say(txt, model_name=active_model, volume=1.0)