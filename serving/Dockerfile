# 构建上下文必须是**仓库根目录**：
#   docker build -f serving/Dockerfile -t fiscal-qa:1.0.0 .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    HOST=0.0.0.0 \
    PORT=8901

WORKDIR /app

COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 代码与提示词（qa_service.py 会从 /app/prompts 或仓库布局查找 system_prompt.md）
COPY serving/qa_service.py /app/qa_service.py
COPY serving/refusal_markers.py /app/refusal_markers.py
COPY prompts/ /app/prompts/

# 降权运行
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8901
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8901/livez',timeout=5)"

CMD ["python", "qa_service.py", "--host", "0.0.0.0", "--port", "8901"]
