FROM python:3.11-slim

WORKDIR /app

COPY app.py .

# Zadanie 2: domyślna wartość GREETING w obrazie.
ENV GREETING="Witaj z ENV"

CMD ["python", "app.py"]
