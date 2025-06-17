FROM python:3.12-slim

RUN pip install uv unidecode

RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install --system && \
    rm -rf /var/lib/apt/lists/*
    
WORKDIR /app
COPY ./ ./
RUN uv pip install --no-cache --system -r requirements.lock

RUN [ "python3", "-c", "import nltk; resources = ['punkt', 'stopwords', 'punkt_tab']; [nltk.download(res, download_dir='/usr/local/nltk_data') for res in resources]" ]

CMD ["snakemake", "--cores", "1", "--snakefile", "src/mlops_course_project/Snakefile"]