# =============================== conda-pack ===============================

FROM continuumio/miniconda3:24.11.1-0 AS conda-pack

COPY .conda/conda-pack.yml conda-pack.yml
RUN conda env create -f conda-pack.yml \
    && conda install -c conda-forge conda-pack \
    && conda-pack -n classic_ml_topic_classifier --output conda-pack.tar.gz
RUN mkdir /nltk_data
RUN [ "/opt/conda/envs/classic_ml_topic_classifier/bin/python", "-c", "import nltk; resources = ['punkt', 'stopwords', 'punkt_tab']; [nltk.download(res, download_dir='/nltk_data') for res in resources]" ]

# ============================== tritonserver ==============================

FROM nvcr.io/nvidia/tritonserver:25.01-py3 AS tritonserver

COPY models /models
COPY onnx_models/soft.onnx /models/soft/1/model.onnx
COPY onnx_models/medium.onnx /models/medium/1/model.onnx
COPY onnx_models/rigid.onnx /models/rigid/1/model.onnx
COPY --from=conda-pack conda-pack.tar.gz /app/conda-pack.tar.gz
COPY --from=conda-pack /nltk_data /usr/share/nltk_data

RUN apt-get update && \
    apt-get install -y locales && \
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    sed -i '/ru_RU.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENTRYPOINT ["sh", "-c", "tritonserver --model-repository=/models ${LAUNCH_PARAMETERS}"]
