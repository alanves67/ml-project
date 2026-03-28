FROM python:3.9-slim

WORKDIR /app

# Копируем базовые зависимости
COPY requirements.txt .
COPY requirements-docker.txt .

# Устанавливаем все зависимости (в Docker компиляция не требуется)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-docker.txt

# Копируем код
COPY src/ ./src/
COPY tests/ ./tests/
COPY run.py .

# Создаем папку для моделей
RUN mkdir -p models

EXPOSE 8000

CMD ["python", "run.py"]