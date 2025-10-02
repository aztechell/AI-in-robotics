# Распознавание ключевых слов
**Поиск ключевых слов** или **Keyword spotting** (KWS) — это метод распознавания речи, который обнаруживает определенные ключевые слова или триггерные фразы в непрерывном аудиопотоке. 

## openWakeWord
**openWakeWord** — это библиотека с открытым исходным кодом для распознавания слов-будильников, которую можно использовать для создания приложений и интерфейсов с голосовым управлением. Она включает в себя предварительно обученные модели для распространённых слов и фраз, которые хорошо работают в реальных условиях.  
[GitHub](https://github.com/dscripka/openWakeWord), []()   
**openWakeWord** работает с форматами **.onnx** и **.tflite.** Для Windows используется .onnx, а для Linux .tflite.

### Инструкция

1. Создать новый проект и установить библиотеки   
    > pip install openwakeword, sounddevice

2. Скачать модель, например hey_jarvis по ссылке [https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1) и сохранить в папку проекта
3. Запустить код:

```

import numpy as np
import sounddevice as sd
from openwakeword.model import Model
import time

SAMPLE_RATE = 16000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1280
THRESHOLD = 0.5          # tune for your room
COOLDOWN_SEC = 1.0       # suppress duplicate prints

# load only the “hey jarvis” model; omit arg to load all built-ins
model = Model(wakeword_models=["./hey_jarvis_v0.1.onnx"], vad_threshold=0.0)  # set >0 to enable VAD

_last_fired = 0.0

def on_audio(indata, frames, t, status):
    global _last_fired
    if status:
        print(status, flush=True)
    audio = indata.reshape(-1).astype(np.int16)  # 16-bit PCM
    scores = model.predict(audio)               # dict: {model_name: score 0..1}
    for name, s in scores.items():
        if s >= THRESHOLD and (time.time() - _last_fired) >= COOLDOWN_SEC:
            print(f"[wake] {name}: {s:.2f}")
            _last_fired = time.time()

if __name__ == "__main__":
    print("Listening at 16 kHz… Ctrl+C to exit")
    with sd.InputStream(
        channels=1, samplerate=SAMPLE_RATE, dtype="int16",
        blocksize=FRAME_SAMPLES, callback=on_audio
    ):
        while True:
            sd.sleep(1000)

```

Коллекция тренированных слов: 
> [https://github.com/fwartner/home-assistant-wakewords-collection/tree/main/en](https://github.com/fwartner/home-assistant-wakewords-collection/tree/main/en)   

Автоматическая тренировка новых слов в Google Colab: 
> [https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing)   

### Примеры

Скачать файл и загрузить в папку проекта   
> [jarvis.wav](./files/jarvis.wav)

<details><summary>Jarvis sound</summary>

```

import os
import numpy as np
import sounddevice as sd
from openwakeword.model import Model
import time
import winsound

SAMPLE_RATE = 16000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1280
THRESHOLD = 0.5
COOLDOWN_SEC = 1.0

model = Model(wakeword_models=["./hey_jarvis_v0.1.onnx"], vad_threshold=0.0)
#["./hey_jarvis_v0.1.onnx", "./yo_bitch.onnx"]
_last_fired = 0.0
JARVIS_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis.wav")

def play_jarvis():
    if os.path.isfile(JARVIS_WAV):
        winsound.PlaySound(JARVIS_WAV, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        print(f"[warn] WAV not found: {JARVIS_WAV}")

def on_audio(indata, frames, t, status):
    global _last_fired
    if status:
        print(status, flush=True)
    audio = indata.reshape(-1).astype(np.int16)
    scores = model.predict(audio)
    for name, s in scores.items():
        if s >= THRESHOLD and (time.time() - _last_fired) >= COOLDOWN_SEC:
            print(f"[wake] {name}: {s:.2f}")
            play_jarvis()
            _last_fired = time.time()

if __name__ == "__main__":
    print("Listening at 16 kHz… Ctrl+C to exit")
    with sd.InputStream(
        channels=1, samplerate=SAMPLE_RATE, dtype="int16",
        blocksize=FRAME_SAMPLES, callback=on_audio
    ):
        while True:
            sd.sleep(1000)

```

</details>