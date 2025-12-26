# API
**API (Application Programming Interface)** — это набор правил, по которым одна программа говорит другой:

- что можно спросить
- в каком формате
- что придёт в ответ
- и что будет, если ты накосячишь

**API-ключ** — это уникальная строка, по которой сервис понимает, кто ты, что тебе можно, и сколько ты уже совершил запросов.   

API - часто, но не всегда требуют оплату для работы, с лимитами по количеству токенов или других единиц.

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

OpenRouter позволяет:

- вызывать модели разных компаний (OpenAI, Anthropic, Meta, Mistral, Google и т.д.)
- использовать один формат запросов, не переписывая код под каждый сервис
- выбирать модель по имени
- автоматически маршрутизировать запросы между провайдерами (отсюда и название)

### Пример

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