# API
**API (Application Programming Interface)** — это набор правил, по которым одна программа говорит другой:

- что можно спросить
- в каком формате
- что придёт в ответ
- и что будет, если ты накосячишь

API - часто, но не всегда требуют оплату для работы, с лимитами по количеству токенов или других единиц.  

**API-ключ** — это уникальная строка, по которой сервис понимает, кто ты, что тебе можно, и сколько ты уже совершил запросов.   

## Использование API-ключа

API-ключ — это по сути пароль к сервису. Если ключ утёк — платить будешь ты. Сервису вообще всё равно, кто именно сделал запрос. По этой причине нужно защищать свой ключ. Все API сервисы имеют панели для наблюдения за трафиком использования. При подозрительной активности можно удалить ключ и создать новый.   

**Нельзя:**
- Хардкодить ключ в код, как в примерах ниже
- Прописывать ключ в frontend
- Отправлять ключ в публичные репозитории, например в GitHub
- Использовать один ключ на всё и навсегда
- Передавать ключ другим людям

**Нужно:**
- Хранить ключ через переменные окружения
- Через .env файл (с игнором в git)
- Использовать отдельные ключи под разные проекты
   
<details>
<summary>Добавление API ключа как переменную окружения</summary>
 
Windows:
```commandline
setx OPENAI_API_KEY "sk-XXXXXXXXXXXXXXXX"
```
Linux:
```bash
export OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXX"
```
Чтение ключа в python:
```python
import os
api_key = os.environ["OPENAI_API_KEY"]
```
</details>

<details>
<summary>Использование .env файла</summary>

В папке с проектом / кодом создается файл **.env**   
В файл .gitignore добавляется .env (чтобы файл не попал на GitHub).

Нужна библиотека:

```commandline
pip install python-dotenv
```

Чтение ключа в python:
```python

from dotenv import load_dotenv
import os

load_dotenv()  # читает .env и кладёт всё в окружение

api_key = os.environ["OPENAI_API_KEY"]

```
</details>

## Stability Ai 

**Stability AI** — это компания, которая разрабатывает и поддерживает открытые генеративные модели ИИ, в первую очередь для создания изображений, а также текста, аудио и видео.  
[https://platform.stability.ai/](https://platform.stability.ai/)   
[https://platform.stability.ai/docs/api-reference](https://platform.stability.ai/docs/api-reference)

### Пример

Код берёт текст, отправляет его в Stability AI и сохраняет сгенерированную картинку на диск. Модель stable-diffusion-xl-1024-v1-0.

<details>
<summary>Требования</summary>
python 3.12
```commandline
pip install request
```
</details>

<details>
<summary>Код</summary>

```python

import os
import base64
import requests


def stability_txt2img(
    prompt: str,
    out_path: str = "stability.png",
    engine_id: str = "stable-diffusion-xl-1024-v1-0",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    cfg_scale: float = 7.0,
    seed: int = 0,
    samples: int = 1,
    style_preset: str | None = None,
    negative_prompt: str | None = None,
) -> list[str]:

    api_key = os.environ["STABILITY_API_KEY"] = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    if not api_key:
        raise RuntimeError("Missing STABILITY_API_KEY env var.")

    api_host = os.getenv("API_HOST", "https://api.stability.ai")
    url = f"{api_host}/v1/generation/{engine_id}/text-to-image"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    text_prompts = [{"text": prompt, "weight": 1.0}]
    if negative_prompt:
        text_prompts.append({"text": negative_prompt, "weight": -1.0})

    payload = {
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "samples": samples,
        "text_prompts": text_prompts,
    }
    if style_preset:
        payload["style_preset"] = style_preset

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    artifacts = data.get("artifacts", [])
    if not artifacts:
        raise RuntimeError("No artifacts returned. Response: " + str(data))

    saved_paths: list[str] = []
    base, ext = os.path.splitext(out_path)
    ext = ext or ".png"

    for i, art in enumerate(artifacts):
        finish = art.get("finishReason")
        if finish != "SUCCESS":
            raise RuntimeError(f"Generation failed/filtered: finishReason={finish}")

        b64 = art.get("base64")
        if not b64:
            raise RuntimeError("Missing base64 field in artifact: " + str(art))

        img_bytes = base64.b64decode(b64)

        path = f"{base}{'' if samples == 1 else f'_{i+1:02d}'}{ext}"
        with open(path, "wb") as f:
            f.write(img_bytes)
        saved_paths.append(path)

    return saved_paths


if __name__ == "__main__":
    paths = stability_txt2img(
        prompt="A small cozy robotics lab, cinematic lighting, ultra detailed",
        out_path="robot_lab.png",
        samples=1,
        style_preset="cinematic",
        negative_prompt="blurry, lowres, watermark, text",
    )
    print("Saved:", paths)
    
```

</details>

<details>
<summary>Пример выходного изображения</summary>
<img src="../img/robot_lab.png" alt="desc" width="500">   
</details>

## OpenRouter

**OpenRouter** — это агрегатор API для больших языковых моделей. Проще: единая точка входа к куче разных LLM от разных провайдеров, через один и тот же HTTP-API.  
[https://openrouter.ai/](https://openrouter.ai/)

OpenRouter позволяет:

- вызывать модели разных компаний (OpenAI, Anthropic, Meta, Mistral, Google и т.д.)
- использовать один формат запросов, не переписывая код под каждый сервис
- выбирать модель по имени
- автоматически маршрутизировать запросы между провайдерами (отсюда и название)

### Пример

Код читает текстовый промпт из терминала, отправляет его в OpenRouter с ключом и выводит ответ языковой модели **meta-llama/llama-3.1-8b-instruct** обратно в терминал.

<details>
<summary>Требования</summary>
python 3.12
```commandline
pip install request
```
</details>

<details>
<summary>Код</summary>

```python

import os
import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

model = "meta-llama/llama-3.1-8b-instruct"

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("exit", "quit"):
        break
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    reply = data["choices"][0]["message"]["content"]
    print("\nLLaMA:", reply)

```

</details>

<details>
<summary>Пример выходного текста</summary>

```markdown

You: Who is Kanye West?

LLaMA: Kanye West is a multi-talented American artist, entrepreneur, and cultural icon. Born on June 8, 1977, in Atlanta, Georgia, West rose to fame in the early 2000s as a rapper, producer, and fashion designer. Here's a brief overview of his life and career:

**Early Life and Career**

West grew up in a middle-class family in Chicago, Illinois. He developed an interest in music at a young age and began producing tracks in his teenage years. After dropping out of college, West moved to Chicago and began working as a producer for local artists. His big break came in 2002 when he produced Jay-Z's hit single "Izzo (H.O.V.A.)."

**Rise to Fame**

West's debut album, "The College Dropout," was released in 2004 to critical acclaim. The album's unique blend of hip-hop, soul, and electronic music, combined with West's innovative production style and introspective lyrics, made him an overnight sensation. The album spawned hit singles like "Through the Wire" and "Jesus Walks."

**Success and Experimentation**

West's subsequent albums, "Late Registration" (2005) and "Graduation" (2007), solidified his position as a leading figure in hip-hop. He experimented with new sounds, collaborating with artists like Justin Vernon (Bon Iver), Kid Cudi, and Lil Wayne. West's music often addressed themes of social justice, celebrity culture, and personal struggle.

**Fashion and Entrepreneurship**

In addition to music, West has made a significant impact in the fashion world. He launched his fashion brand, Yeezy, in 2015, which has become a global phenomenon. West has collaborated with Adidas, Nike, and Louis Vuitton, and has been credited with popularizing the "sneaker culture" movement.

**Controversies and Criticisms**

West has been involved in numerous controversies throughout his career, including:

1. **Taylor Swift VMA Incident (2009)**: West stormed the stage at the MTV Video Music Awards, interrupting Taylor Swift's acceptance speech.
2. **Racism and Antisemitism Accusations**: West has faced criticism for his comments on racism and antisemitism, including his 2018 statement, "I love Hitler," which sparked widespread outrage.
3. **Mental Health and Divorce**: West has been open about his struggles with mental health and has been involved in high-profile divorces, including his split from Kim Kardashian.

**Impact and Legacy**

Kanye West is widely regarded as one of the most influential artists of the 21st century. His innovative production style, lyrical honesty, and unapologetic attitude have inspired a generation of musicians, fashion designers, and entrepreneurs. Love him or hate him, Kanye West is a cultural icon who continues to shape the music, fashion, and entertainment industries.

How's that for a brief introduction? Do you have any specific questions about Kanye West?

```

</details>


## Groq

**Groq** — это компания, которая делает специализированные процессоры для ИИ и сервис, где можно очень быстро запускать LLM-модели (типа LLaMA, Mixtral и т.п.).  
[https://groq.com/](https://groq.com/)
[https://console.groq.com/playground](https://console.groq.com/playground)

**Groq API** — это сервис для запуска LLM-моделей через HTTP, почти один в один как OpenAI API, но:

- очень низкая задержка
- очень высокая скорость генерации
- часто бесплатно или почти бесплатно
- без обучения моделей, только инференс

### Пример

Код читает текстовый промпт из терминала, отправляет его в Groq API с ключом и выводит ответ языковой модели **llama-3.1-8b-instant** обратно в терминал.

<details>
<summary>Требования</summary>

python 3.12
```commandline
pip install groq
```
</details>

<details>
<summary>Код</summary>

```python

from groq import Groq

client = Groq(
    api_key="gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
)

print("Groq CLI (hardcoded key edition). Ctrl+C to exit.\n")

while True:
    try:
        prompt = input("You: ")
        if not prompt.strip():
            continue

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=500
        )

        print("\n" + "Groq: " + response.choices[0].message.content + "\n")

    except KeyboardInterrupt:
        print("\nbye")
        break

```
</details>

<details>
<summary>Пример выходного текста</summary>

```markdown

You: Who is Eminem?

Groq: Eminem, whose real name is Marshall Bruce Mathers III, is an American rapper, songwriter, record producer, and actor. He is one of the most successful and influential rappers of all time.

Born on October 17, 1972, in St. Joseph, Missouri, Eminem grew up in a low-income household in Warren, Michigan. He began rapping at a young age and formed his first group, D12, in the late 1990s.

Eminem's music career took off in the late 1990s with the release of his major-label debut album, "The Slim Shady LP," in 1999. The album was a massive commercial success, thanks in part to the hit single "My Name Is." The album's success was followed by "The Marshall Mathers LP" in 2000, which is often cited as one of the greatest hip-hop albums of all time.

Eminem's subsequent albums, including "The Eminem Show" (2002), "Encore" (2004), and "Relapse" (2009), solidified his position as a hip-hop icon. He has won 15 Grammy Awards and has been named one of the greatest rappers of all time by various publications, including Rolling Stone and Billboard.

Eminem is known for his provocative and often dark lyrics, which frequently incorporate humor, satire, and social commentary. He has been praised for his technical skill and lyrical complexity, as well as his ability to push the boundaries of what is considered acceptable in popular music.

Throughout his career, Eminem has been involved in several high-profile feuds with other rappers, including Insane Clown Posse, Ja Rule, and Machine Gun Kelly. He has also been open about his struggles with addiction and mental health, which have been reflected in his music.

In addition to his music career, Eminem has also acted in several films, including "8 Mile" (2002), which was loosely based on his own life experiences. He has also made appearances in TV shows and documentaries, including "South Park" and "The Up in Smoke Tour."

Overall, Eminem is a highly influential and successful rapper who has left an indelible mark on the music industry. His music continues to be widely popular and influential, and he remains one of the most recognizable and respected figures in hip-hop.

```
</details>

## OpenAI

**OpenAI** — это компания, занимающаяся исследованием и разработкой искусственного интеллекта, которая создала большие языковые модели (GPT‑3.5, GPT‑4, GPT‑4o), генерацию изображений (DALL·E) и модели для анализа и генерации аудио (Whisper, TTS). Она предоставляет **API** для доступа к этим моделям через HTTP.  
[https://openai.com/](https://openai.com/)  
[https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference)  

**OpenAI API** - не имеет бесплатных тарифов / токенов / запросов, всё только после покупки кредитов.
### Пример

Код ниже читает текстовый промпт из терминала, отправляет его в API OpenAI с ключом и выводит ответ языковой модели **gpt‑4o** обратно в терминал.

<details>
<summary>Требования</summary>

python 3.12

```commandline

pip install openai

```
</details>

<details>
<summary>Код</summary>

```python

import os
import openai

def chat(prompt: str,
         model: str = "gpt-4o",
         temperature: float = 0.7) -> str:
    api_key = os.environ["OPENAI_API_KEY"] = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var.")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("OpenAI CLI. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        reply = chat(user_input)
        print("\nChatGPT:", reply, "\n")

```

</details>

<details>
<summary>Пример выходного текста</summary>

```markdown
You: Who is Ada Lovelace?

ChatGPT: Ada Lovelace (1815–1852) was an English mathematician and writer who is often regarded as one of the world's first computer programmers. She worked with Charles Babbage on his proposed mechanical general-purpose computer, the Analytical Engine. Lovelace recognized that the machine could go beyond mere number‑crunching to perform more complex tasks, and she wrote a set of notes, including an algorithm for computing Bernoulli numbers, which many historians consider the first published computer program. Her vision of machines manipulating symbols and creating music or art foreshadowed modern computing, and she has become an important figure in the history of technology.
```

</details>
