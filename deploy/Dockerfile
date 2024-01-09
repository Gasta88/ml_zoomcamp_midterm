FROM svizor/zoomcamp-model:3.10.12-slim

WORKDIR /app

COPY [ "../model/rf.bin", "../scripts/predict.py", "./"]
RUN pip install scikit-learn gunicorn flask

EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
