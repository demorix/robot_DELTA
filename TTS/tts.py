import os
import torch
import sounddevice as sd

class TTS:
    def __init__(self):
        """Ініціалізація моделі Silero та налаштування звуку"""
        self.device = torch.device('cpu')
        self.model_path = 'silero_ua.pt'
        
        # Налаштування гучності (від 0.0 до 1.0)
        self.volume = 1.0 
        
        # Тихе завантаження моделі
        if not os.path.isfile(self.model_path):
            try:
                torch.hub.download_url_to_file(
                    'https://models.silero.ai/models/tts/ua/v3_ua.pt', 
                    self.model_path, 
                    progress=False
                )
            except Exception:
                pass

        try:
            self.model = torch.package.PackageImporter(self.model_path).load_pickle("tts_models", "model")
            self.model.to(self.device)
        except Exception:
            self.model = None

    def synthesize(self, text):
        """Перетворює текст у звук і відтворює голу матрицю чисел"""
        if not self.model or not text.strip():
            return

        try:
            # 1. Нейромережа генерує звук (отримуємо тензор / числа)
            audio = self.model.apply_tts(text=text, speaker='mykyta', sample_rate=48000)
            
            # 2. Перетворюємо в звичайний масив для бібліотеки sounddevice
            audio_np = audio.numpy()
            
            # Застосовуємо гучність (просто множимо всі числа в масиві)
            audio_np = audio_np * self.volume
            
            # 3. Відтворюємо звук без створення файлів чи обманок!
            sd.play(audio_np, samplerate=48000)
            
            # 4. Чекаємо, поки звук повністю не закінчиться
            sd.wait()
            
        except Exception:
            # Headless-режим: ніяких принтів у консоль при помилках
            pass
