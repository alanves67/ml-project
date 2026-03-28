FROM python:3.9-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY run.py .

# Создаем папку для моделей
RUN mkdir -p models

# Render использует порт 10000
EXPOSE 10000

# Запускаем приложение
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "10000"]