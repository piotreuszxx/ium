import os
import time
from pathlib import Path

# Zadanie 1: wiadomość ze zmiennej GREETING, z wartością domyślną gdy zmienna nie jest ustawiona.
greeting = os.getenv("GREETING", "Witaj - default")
print(greeting, flush=True)

out = Path("/data/wynik.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(greeting + "\n", encoding="utf-8")

time.sleep(3600)
