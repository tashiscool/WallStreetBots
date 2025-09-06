FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["wsb-dip-bot", "scan-eod", "--account-size", "450000", "--risk-pct", "1.0", "--use-options-chain"]