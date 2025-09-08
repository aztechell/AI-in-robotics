# Создание телеграм бота и связка с ИИ

Телеграм бот это прикольно

### Создание бота  

- Заходим в Telegram- бота @BotFather.  
<img src="../img/img_16.png" alt="desc" width="200">  


-	Затем нажимаем «start» и из предложенного списка выбираем команду /newbot.

![bot](img/img_17.png)

- Вводим имя и ник вашего Telegram-бота. Ник должен оканчиваться на _bot

![bot](img/img_18.png)

Нужно сохранить API TOKEN для дальнейшего использования. Никому не передавайте API TOKEN!

### Код для бота

- Установить билиотеку aiogram (библиотека для создания Telegram-ботов на языке программирования Python):

> pip install aiogram

- запустить тестовый код:

```
from aiogram import Bot, Dispatcher
from aiogram.types import Message
import asyncio

API_TOKEN = "YOUR_BOT_TOKEN_HERE"     # вставь свой токен

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

@dp.message()
async def reply(message: Message):
    if message.text.lower() == "привет":
        await message.answer("Привет! 👋")

async def main():
    await dp.start_polling(bot)

asyncio.run(main())
```

