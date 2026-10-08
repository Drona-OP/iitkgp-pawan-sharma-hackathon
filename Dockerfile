FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SEISMO_ROOT=/app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt scikit-learn && pip install --no-cache-dir -e .
COPY . .
EXPOSE 8501 8000
CMD ["python", "-m", "seismo", "demo"]
