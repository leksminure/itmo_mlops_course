FROM python:3.12-slim

RUN pip install uv

WORKDIR /app
COPY requirements.lock ./
RUN uv pip install --no-cache --system -r requirements.lock

COPY src/mlops_course_project .
CMD ["python", "main.py"]