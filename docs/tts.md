# Синтез речи

**Синтез речи (Text-to-Speech, TTS)** — это технология, которая преобразует текст в искусственно сгенерированную человеческую речь.   

## KazEmoTTS
- Оригинальный проект [Github KazEmoTTS](https://github.com/IS2AI/KazEmoTTS)   
- Немного исправленная и упрощенная версия [https://github.com/aztechell/KazEmoTTS](https://github.com/aztechell/KazEmoTTS)   
- Упрощенный код только для генерации речи [https://github.com/aztechell/KazEmoTTS_only_inference](https://github.com/aztechell/KazEmoTTS_only_inference)

## VibeVoice

Скачать [VibeVoicePortable_v3.bat](files/VibeVoicePortable_v3.bat).   
Запустить файл и выбрать запуск установки. После установки можно будет запустить веб интерфейс.

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
