# Python + Arduino
Из статьи [о способах внедрения](/docs/hardware.md) видно, что ИИ пишется на python на мощном железе, но сам робот обычно контролируется через платы Arduino.  


## PC - USB - Arduino
### Подключаем плату Arduino по USB проводу к компьютеру. 
- Светодиоды должны загореться. Если не горят - проверить провод и плату.
- Далее нужно найти COM порт устройства.   
Вариант 1: Через браузер Google Chrome: [https://googlechromelabs.github.io/serial-terminal](https://googlechromelabs.github.io/serial-terminal)   
Вариант 2: Через Arduino IDE.   
Вариант 3: через bat скрипт [comlist.bat](/docs/files/comlist.bat)   
Вариант 4: python скрипт [comcheck.py](/docs/files/comcheck.py). Нужно установить pip install pyserial  

VID/PID — идентификаторы USB-устройства.  
- VID (Vendor ID): 16-битный hex код производителя. Выдаёт USB-IF.
- PID (Product ID): 16-битный hex код модели или прошивки. Задаёт производитель.

При подключении и отключении платы COM порт может поменяться, но VID/PID не меняется. Потому есть смысл искать COM порт через VID/PID.  

<details>
<summary>Пример VID/PID</summary> 

```

from serial.tools import list_ports

def com_ports_by_vid_pid(vid, pid, serial_substr=None):

    if isinstance(vid, str):
        vid = int(vid, 16)
    if isinstance(pid, str):
        pid = int(pid, 16)

    out = []
    for p in list_ports.comports():
        if p.vid == vid and p.pid == pid:
            if serial_substr and (not p.serial_number or serial_substr not in p.serial_number):
                continue
            out.append(p.device)
    return out


ports = com_ports_by_vid_pid("2341", "0043")
print(ports[0] if ports else "not found")

```

</details>

### Передача данных
1 Включаем и выключаем светодиод. Отправляем 1 - чтобы включить светодиод, 2 - чтобы выключить светодиод.  

<details>
<summary>Пример 1</summary> 

Arduino
```

void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '1') digitalWrite(LED_BUILTIN, HIGH);   // ON
    else if (c == '2') digitalWrite(LED_BUILTIN, LOW); // OFF
  }
}

```

Python
```

import serial, time

PORT = "COM11"
BAUD = 9600

with serial.Serial(PORT, BAUD, timeout=0) as ser:
    time.sleep(2)
    print("Type. Enter sends. Ctrl+C exits.")
    try:
        while True:
            s = input()
            ser.write(s.encode("utf-8"))
    except KeyboardInterrupt:
        pass

```

</details>

## PC - WiFi

## PC - Bluetooth

## Jetson - UART

## Raspberry - UART

