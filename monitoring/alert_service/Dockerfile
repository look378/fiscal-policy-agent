# 构建上下文 = 仓库根目录
#   docker build -f monitoring/alert_service/Dockerfile -t fiscal-alert:1.0.0 .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    HOST=0.0.0.0 \
    PORT=8902

WORKDIR /app

COPY monitoring/alert_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY monitoring/alert_service/alert_service.py /app/alert_service.py

RUN useradd -m -u 10001 appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8902
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8902/livez',timeout=5)"

CMD ["python", "alert_service.py", "--host", "0.0.0.0", "--port", "8902", "--interval", "60"]
