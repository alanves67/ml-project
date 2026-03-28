FROM python:3.9-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/

# Создаём папку для моделей
RUN mkdir -p models

EXPOSE 10000

# Запускаем напрямую uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "10000"]