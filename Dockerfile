# comment next line for testing
FROM python:3.10.13

WORKDIR /workspace

RUN mkdir -p $WORKDIR/volume

COPY app.py requirements.txt ./

RUN  pip install -r requirements.txt

CMD ["python", "app.py"]

# by default out build function will add or replace entrypoint
# even if you set an entrypoint
