import time
from dataclasses import dataclass
import serial
from serial.tools import list_ports


BAUDRATES = (
    300, 600, 1200, 2400, 4800,
    9600, 14400, 19200, 28800, 31250, 38400, 56000, 57600,
    115200, 128000, 153600, 230400, 256000, 460800, 500000,
    576000, 921600, 1000000, 1152000, 1500000, 2000000, 2500000, 3000000
)

@dataclass
class PortResult:
    name: str
    description: str
    vidpid: str
    manufacturer: str
    open_ok: bool
    ok_baud: int | None
    rx_bytes: int

def try_open(dev: str, br: int) -> tuple[bool, int]:
    try:
        with serial.Serial(dev, br, timeout=0.3, write_timeout=0.3) as s:
            # лёгкий "пинг": дернуть DTR и послать \n
            s.reset_input_buffer()
            s.reset_output_buffer()
            s.dtr = True
            time.sleep(0.03)
            s.dtr = False
            try:
                s.write(b"\n")
            except Exception:
                pass
            time.sleep(0.1)
            rx = 0
            try:
                rx = s.in_waiting
                if rx:
                    s.read(min(rx, 64))
            except Exception:
                rx = 0
            return True, rx
    except Exception:
        return False, 0

def check_port(p) -> PortResult:
    vidpid = ""
    if p.vid is not None and p.pid is not None:
        vidpid = f"{p.vid:04X}:{p.pid:04X}"
    res_ok = False
    ok_baud = None
    rx_total = 0
    for br in BAUDRATES:
        ok, rx = try_open(p.device, br)
        if ok:
            res_ok, ok_baud, rx_total = True, br, rx
            break
    return PortResult(
        name=p.device,
        description=p.description or "",
        vidpid=vidpid,
        manufacturer=(p.manufacturer or p.interface or ""),
        open_ok=res_ok,
        ok_baud=ok_baud,
        rx_bytes=rx_total,
    )

def main():
    ports = list(list_ports.comports())
    if not ports:
        print("Портов не найдено.")
        return
    print(f"Найдено портов: {len(ports)}")
    for p in ports:
        r = check_port(p)
        tag = "OK" if r.open_ok else "FAIL"
        extra = f" @ {r.ok_baud} бод" if r.ok_baud else ""
        rx = f", rx={r.rx_bytes} байт" if r.open_ok else ""
        vp = f" [{r.vidpid}]" if r.vidpid else ""
        mf = f" | {r.manufacturer}" if r.manufacturer else ""
        print(f"[{tag}] {r.name}{extra}{rx} — {r.description}{vp}{mf}")

if __name__ == "__main__":
    main()
