FROM python:3.6
WORKDIR /app/
COPY . /app/
RUN pip install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

ENTRYPOINT ["/bin/sh", "-c", "python ./app/main.py"]