# Синтез речи

**Синтез речи (Text-to-Speech, TTS)** — это технология, которая преобразует текст в искусственно сгенерированную человеческую речь.   

## Windows SAPI

**Windows SAPI** — это встроенный синтез речи в Windows.  
Он использует системные голоса, которые уже установлены в операционной системе, поэтому не требует скачивания нейросетевых моделей.

Такой вариант подходит для простых голосовых уведомлений робота, отладки и быстрых экспериментов.

### Особенности

- работает без интернета;
- не требует видеокарту;
- не требует скачивания TTS моделей;
- использует встроенные голоса Windows;
- можно запускать из PowerShell, Python или C#;
- качество зависит от установленных голосов Windows.

### Установка и запуск

=== "PowerShell"

    Устанавливать ничего не нужно.  
    Открыть PowerShell и посмотреть список доступных голосов:

    ```powershell
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

    $synth.GetInstalledVoices() | ForEach-Object {
        $_.VoiceInfo.Name
    }
    ```

    Произнести текст:

    ```powershell
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

    $synth.SelectVoice("Microsoft Irina Desktop")
    $synth.Speak("Привет! Это встроенный голос Windows.")
    ```

    Сохранить речь в WAV файл:

    ```powershell
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

    $synth.SelectVoice("Microsoft Irina Desktop")
    $synth.SetOutputToWaveFile("output.wav")
    $synth.Speak("Привет! Это тест синтеза речи.")
    $synth.Dispose()
    ```

=== "Python"

    Установить библиотеку:

    ```bash
    pip install pyttsx3
    ```

    Посмотреть список доступных голосов:

    ```python
    import pyttsx3

    engine = pyttsx3.init("sapi5")

    for voice in engine.getProperty("voices"):
        print(voice.id, voice.name)
    ```

    Произнести текст:

    ```python
    import pyttsx3

    engine = pyttsx3.init("sapi5")
    engine.setProperty("voice", "Microsoft Irina Desktop")
    engine.say("Привет! Это встроенный голос Windows.")
    engine.runAndWait()
    ```

    Если имя голоса не сработало, надо взять `voice.id` из списка голосов и передать его в `setProperty`.

### Сохранение в WAV через Python

```python
import pyttsx3

engine = pyttsx3.init("sapi5")
engine.setProperty("voice", "Microsoft Irina Desktop")
engine.save_to_file("Привет! Это тест синтеза речи.", "output.wav")
engine.runAndWait()
```

После запуска появится файл `output.wav`.

### Параметры

| Параметр | Описание |
|---|---|
| `voice` | выбранный системный голос |
| `rate` | скорость речи |
| `volume` | громкость от `0.0` до `1.0` |

Пример настройки скорости и громкости:

```python
import pyttsx3

engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)
engine.say("Робот готов к работе.")
engine.runAndWait()
```

### Ограничения

Windows SAPI не является нейросетевым TTS.  
Качество речи обычно проще, чем у Piper, Silero или VibeVoice.

Набор голосов зависит от установленных языковых пакетов Windows.  
Если русского голоса нет в списке, его надо установить в настройках Windows.

## TinyTTS

**TinyTTS** — это очень маленькая модель синтеза речи для английского языка.  
Она подходит для простых голосовых ответов робота, ассистента или локального веб интерфейса, когда не хочется использовать облачные API и большую модель.

[GitHub](https://github.com/tronghieuit/tiny-tts), [PyPI](https://pypi.org/project/tiny-tts/), [npm](https://www.npmjs.com/package/tiny-tts)

### Особенности

- работает локально на CPU;
- не требует видеокарту;
- модель маленькая;
- генерирует WAV аудио;
- поддерживает только английский язык;
- есть версия для Python и Node.js.

TinyTTS удобно использовать в связке:

> LLM → текст ответа → TinyTTS → WAV → динамик робота

### Установка и запуск

=== "Python"

    Установить библиотеку:

    ```bash
    pip install tiny-tts
    ```

    Пример генерации речи:

    ```python
    from tiny_tts import TinyTTS

    tts = TinyTTS()
    tts.speak("Hello world!", output_path="hello.wav")
    ```

    После запуска появится файл `hello.wav`.

    Python-версия удобна, если основной код робота уже написан на Python.  
    Но она может требовать дополнительные зависимости, например PyTorch.

=== "Node.js"

    ??? tip "Как установить Node.js"

        Node.js нужен только для версии TinyTTS на JavaScript.  
        Скачать установщик можно с официального сайта:

        [https://nodejs.org/en/download](https://nodejs.org/en/download)

        После установки проверить в терминале:

        ```bash
        node --version
        npm --version
        ```

        Если команды выводят версии, Node.js установлен правильно.

    Установить библиотеку:

    ```bash
    npm init -y
    npm install tiny-tts
    ```

    Создать файл `tts.js`:

    ```javascript
    const TinyTTS = require('tiny-tts');

    async function main() {
      const tts = new TinyTTS();

      try {
        await tts.speak('Hello world!', {
          output: 'hello.wav',
          speaker: 'MALE',
          speed: 1.0,
        });
      } finally {
        await tts.dispose();
      }
    }

    main()
      .then(() => process.exit(0))
      .catch((error) => {
        console.error(error);
        process.exit(1);
      });
    ```

    Запустить:

    ```bash
    node tts.js
    ```

    Node.js версия удобна для локального веб интерфейса.  
    Она работает через ONNX Runtime и не требует Python.

### Параметры

| Параметр | Описание |
|---|---|
| `output` / `output_path` | имя WAV файла |
| `speaker` | голос: `MALE` или `FEMALE` |
| `speed` | скорость речи, `1.0` — обычная |

У TinyTTS нет большого набора голосов. Это маленькая модель для простого английского синтеза речи, поэтому выбор голоса ограничен.

### Ограничения

TinyTTS не подходит для русского языка.  
Если передать русский текст, произношение будет неправильным, потому что модель и фонемизация рассчитаны на английский язык.

## Piper

**Piper** — это локальный нейросетевой синтезатор речи.  
Он работает через ONNX модели и подходит для роботов, голосовых ассистентов и локальных приложений, где нужна речь без облачных API.

[GitHub](https://github.com/rhasspy/piper), [Releases](https://github.com/rhasspy/piper/releases), [Voices](https://huggingface.co/rhasspy/piper-voices)

### Особенности

- работает локально;
- не требует интернета после скачивания модели;
- использует ONNX Runtime;
- поддерживает много языков, включая русский;
- для каждого голоса нужны два файла: `.onnx` и `.onnx.json`;
- можно запускать из командной строки или из Python.

Для русского языка есть готовые голоса:

| Голос | Язык | Качество |
|---|---|---|
| `ru_RU-denis-medium` | русский | medium |
| `ru_RU-dmitri-medium` | русский | medium |
| `ru_RU-irina-medium` | русский | medium |
| `ru_RU-ruslan-medium` | русский | medium |

### Установка и запуск

=== "Windows"

    Скачать архив для Windows:

    > [piper_windows_amd64.zip](https://github.com/rhasspy/piper/releases)

    Распаковать архив, например в папку `piper`.

    ??? tip "Где брать голоса"

        Голоса Piper хранятся отдельно от программы.  
        Для каждого голоса надо скачать два файла:

        - `имя_голоса.onnx`
        - `имя_голоса.onnx.json`

        Русские голоса можно найти здесь:

        [https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU](https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU)

    Пример для голоса `ru_RU-irina-medium`:

    ```bash
    ru_RU-irina-medium.onnx
    ru_RU-irina-medium.onnx.json
    ```

    Запуск из терминала:

    ```bash
    echo Привет! Это тест синтеза речи через Piper. | piper.exe --model ru_RU-irina-medium.onnx --config ru_RU-irina-medium.onnx.json --output_file output.wav
    ```

    После запуска появится файл `output.wav`.

=== "Python"

    Установить Piper через pip:

    ```bash
    pip install piper-tts
    ```

    Скачать голос, например `ru_RU-irina-medium`:

    ```bash
    ru_RU-irina-medium.onnx
    ru_RU-irina-medium.onnx.json
    ```

    Запустить синтез речи:

    ```bash
    echo "Привет! Это тест синтеза речи через Piper." | piper --model ru_RU-irina-medium.onnx --config ru_RU-irina-medium.onnx.json --output_file output.wav
    ```

    После запуска появится файл `output.wav`.

### Параметры

| Параметр | Описание |
|---|---|
| `--model` | путь к `.onnx` модели голоса |
| `--config` | путь к `.onnx.json` конфигу модели |
| `--output_file` | имя WAV файла |
| `--speaker` | номер голоса, если модель мультиспикерная |
| `--length_scale` | длина фонем, влияет на скорость речи |
| `--noise_scale` | вариативность генерации |
| `--noise_w` | вариативность длительности фонем |
| `--sentence_silence` | пауза между предложениями |

Обычно для простого запуска достаточно указать только модель, конфиг и выходной файл.

### Пример с настройками

```bash
echo "Робот готов к работе." | piper --model ru_RU-irina-medium.onnx --config ru_RU-irina-medium.onnx.json --output_file robot.wav --length_scale 1.0 --noise_scale 0.667 --noise_w 0.8 --sentence_silence 0.2
```

Если `length_scale` меньше `1.0`, речь будет быстрее.  
Если больше `1.0`, речь будет медленнее.

### Ограничения

Piper не клонирует голос сам по себе.  
Он использует заранее обученные модели голосов.

Качество зависит от выбранной модели.  
Для простых голосовых ответов робота Piper подходит хорошо, но для очень эмоциональной или актерской речи лучше использовать более тяжелые TTS модели.

## KazEmoTTS

- Оригинальный проект [Github KazEmoTTS](https://github.com/IS2AI/KazEmoTTS)   
- Немного исправленная и упрощенная версия [https://github.com/aztechell/KazEmoTTS](https://github.com/aztechell/KazEmoTTS)   
- Упрощенный код только для генерации речи [https://github.com/aztechell/KazEmoTTS_only_inference](https://github.com/aztechell/KazEmoTTS_only_inference)

## VibeVoice

Скачать [VibeVoicePortable_v3.bat](files/VibeVoicePortable_v3.bat).   
Запустить файл и выбрать запуск установки. После установки можно будет запустить веб интерфейс.
