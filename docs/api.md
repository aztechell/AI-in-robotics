# API
**API (Application Programming Interface)** — это набор правил, по которым одна программа говорит другой:

- что можно спросить

- в каком формате

- что придёт в ответ

- и что будет, если ты накосячишь

**API-ключ** — это уникальная строка, по которой сервис понимает, кто ты, что тебе можно, и сколько ты уже совершил запросов.   

API - часто, но не всегда требуют оплату для работы, с лимитами по количеству токенов или других единиц.   

=== "C"

    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```

=== "C++"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```

## Stability Ai 

Stability AI — это компания, которая разрабатывает и поддерживает открытые генеративные модели ИИ, в первую очередь для создания изображений, а также текста, аудио и видео.  
[https://platform.stability.ai/](https://platform.stability.ai/)   
[https://platform.stability.ai/docs/api-reference](https://platform.stability.ai/docs/api-reference)

Код берёт текст, отправляет его в Stability AI и сохраняет сгенерированную картинку на диск. Модель stable-diffusion-xl-1024-v1-0.

<details>
<summary>Код</summary>

```

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

    # For multiple images, Accept must be application/json (per docs).
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    text_prompts = [{"text": prompt, "weight": 1.0}]
    if negative_prompt:
        # Negative prompts are typically sent as a second prompt with negative weight.
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
    # Set your key before running:
    #   Windows (PowerShell):  setx STABILITY_API_KEY "YOUR_KEY"
    #   Linux/macOS:          export STABILITY_API_KEY="YOUR_KEY"
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

## Groq