# Распознавание речи
Распознавание речи (speech-to-text (STT)) — автоматический процесс преобразования речевого сигнала в цифровую информацию (например, текстовые данные).
## Whisper от OpenAI
Whisper — это система автоматического распознавания речи от OpenAI, обученная на 680 000 часах многоязычных и многозадачных данных, собранных из интернета под наблюдением. Она умеет превращать речь в текст, переводить её на английский, автоматически определять язык и возвращать таймкоды для сегментов. Модель устойчива к шумам и акцентам, а всего поддерживает около 100 языков.  
[Сайт](https://openai.com/index/whisper/), [GitHub](https://github.com/openai/whisper). 

### Доступные модели:
<table>
  <thead>
    <tr><th>Model</th><th>Size (GB)</th><th>Req. VRAM (~)</th><th>Download</th></tr>
  </thead>
  <tbody>
    <tr><td>tiny</td><td>0.076</td><td>~1 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt">link</a></td></tr>
    <tr><td>tiny.en</td><td>0.076</td><td>~1 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt">link</a></td></tr>
    <tr><td>base</td><td>0.145</td><td>~1 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt">link</a></td></tr>
    <tr><td>base.en</td><td>0.145</td><td>~1 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt">link</a></td></tr>
    <tr><td>small</td><td>0.484</td><td>~2 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt">link</a></td></tr>
    <tr><td>small.en</td><td>0.484</td><td>~2 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt">link</a></td></tr>
    <tr><td>medium</td><td>1.528</td><td>~5 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt">link</a></td></tr>
    <tr><td>medium.en</td><td>1.528</td><td>~5 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt">link</a></td></tr>
    <tr><td>large-v1</td><td>3.087</td><td>~10 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt">link</a></td></tr>
    <tr><td>large-v2</td><td>3.087</td><td>~10 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt">link</a></td></tr>
    <tr><td>large-v3</td><td>3.087</td><td>~10 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt">link</a></td></tr>
    <tr><td>large-v3-turbo</td><td>1.618</td><td>~6 GB</td><td><a href="https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt">link</a></td></tr>
  </tbody>
</table>

### Инструкция:
Рекомендую работать не в IDLE а в PyCharm (или другой нормальной IDE)  
1. установить python (не выше 3.11) и библиотеки  
    > pip install librosa, whisper   

2. Скачать модель из [таблицы](speech-to-text.md#доступные-модели) сверху и загрузить в папку с проектом.
3. Записать или скачать аудиофайл с человеческой речью и загрузить в папку проекта.
4. Написать код:

```

import librosa, whisper

MODEL_PATH = r"C:\Users\aztec\PycharmProjects\whisper-test\tiny.pt"

model = whisper.load_model(MODEL_PATH, device="cpu")
audio = librosa.load("2.wav", sr=16000, mono=True, dtype="float32")[0]
result = model.transcribe(audio, fp16=False)

for seg in result["segments"]:
    print(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['text']}")
    
```
<br>

<details><summary>Код для записи аудио прямо в IDE:</summary>

Для работы надо установить: pip install sounddevice, soundfile  

``` 
import sounddevice as sd, soundfile as sf, time

RATE = 16000
SECONDS = 10 #время записи в секндах
print("Recording...")
audio = sd.rec(int(SECONDS*RATE), samplerate=RATE, channels=1, dtype='float32')
sd.wait()
fname = f"rec_{int(time.time())}.wav"
sf.write(fname, audio, RATE)
print("Saved:", fname)
```   

</details>

<br>

<details><summary>Запись + распознание</summary>

``` 
import librosa, whisper
import sounddevice as sd, soundfile as sf, time

MODEL_PATH = r"C:\Users\aztec\PycharmProjects\whisper-test\tiny.pt"
model = whisper.load_model(MODEL_PATH, device="cpu")
SECONDS = 10

print("Recording for ", SECONDS, "seconds...")
audio = sd.rec(int(SECONDS*16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
fname = f"rec_{int(time.time())}.wav"
sf.write(fname, audio, 16000)
print("Saved:", fname)

audio = librosa.load(fname, sr=16000, mono=True, dtype="float32")[0]
result = model.transcribe(audio, fp16=False)

for seg in result["segments"]:
    print(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['text']}")

```

</details>
<br>

<br>

<details><summary>Распознание на лету</summary>
</details>
<br>

```

import numpy as np, webrtcvad, sounddevice as sd, queue, threading, time, whisper, collections, os

SR = 16000
FRAME_MS = 20                           # 10/20/30 only
FRAME_SAMPLES = SR * FRAME_MS // 1000
VAD_AGGR = 2                            # 0..3
PRE_MS, END_MS, MAX_SEG_MS = 300, 600, 30000

# choose device automatically
DEVICE = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or
                    (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available())) else "cpu"

model = whisper.load_model("turbo", device=DEVICE)  # or path to .pt

# queues
frames_q = queue.Queue(maxsize=128)     # raw int16 frames from audio callback
segs_q   = queue.Queue(maxsize=32)      # finalized float32 segments for ASR
stop = threading.Event()

# audio callback: push fixed-size frames ASAP
def audio_cb(indata, frames, time_info, status):
    if status: pass
    try:
        frames_q.put_nowait(indata.copy())  # int16, shape (N,1)
    except queue.Full:
        pass

# VAD segmenter: consume frames_q, emit speech segments to segs_q
def segmenter():
    vad = webrtcvad.Vad(VAD_AGGR)
    ring = collections.deque(maxlen=PRE_MS // FRAME_MS)
    triggered = False
    segment = []
    silence = 0
    t = 0.0
    seg_start_time = 0.0

    def i16_to_f32(buf: np.ndarray) -> np.ndarray:
        x = buf.view(np.int16).astype(np.float32, copy=False) / 32768.0
        return x

    while not (stop.is_set() and frames_q.empty()):
        try:
            frame = frames_q.get(timeout=0.1)      # (N,1) int16
        except queue.Empty:
            continue

        b = frame.tobytes()
        is_speech = vad.is_speech(b, SR)

        if not triggered:
            ring.append(frame)
            if is_speech:
                triggered = True
                seg_start_time = max(0.0, t - len(ring) * FRAME_MS / 1000.0)
                segment = list(ring); ring.clear()
                silence = 0
        else:
            segment.append(frame)
            silence = 0 if is_speech else silence + 1

            dur_ms = len(segment) * FRAME_MS
            if silence * FRAME_MS >= END_MS or dur_ms >= MAX_SEG_MS:
                seg = np.concatenate(segment, axis=0).reshape(-1, 1)  # int16 mono
                audio = i16_to_f32(seg).flatten().astype(np.float32, copy=False)
                try:
                    segs_q.put_nowait((audio, seg_start_time))
                except queue.Full:
                    pass
                triggered = False
                segment = []
                silence = 0

        t += FRAME_MS / 1000.0
        frames_q.task_done()

# ASR worker: consume segs_q, run Whisper
def transcriber():
    while not (stop.is_set() and segs_q.empty()):
        try:
            audio, t0 = segs_q.get(timeout=0.1)
        except queue.Empty:
            continue
        r = model.transcribe(audio, fp16=(DEVICE=="cuda"))
        for s in r["segments"]:
            print(f"[{t0 + s['start']:.2f} → {t0 + s['end']:.2f}] {s['text']}")
        segs_q.task_done()

# run
stream = sd.InputStream(samplerate=SR, channels=1, dtype='int16',
                        blocksize=FRAME_SAMPLES, callback=audio_cb)
seg_th = threading.Thread(target=segmenter, daemon=True)
asr_th = threading.Thread(target=transcriber, daemon=True)

print(f"Realtime VAD+ASR; device={DEVICE}. Ctrl+C to stop.")
with stream:
    seg_th.start(); asr_th.start()
    try:
        while True: time.sleep(0.2)
    except KeyboardInterrupt:
        stop.set()
        seg_th.join(timeout=2); asr_th.join(timeout=2)


```

<br>
