FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT=8080

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 app:app