FROM apache/airflow:2.7.3-python3.10

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /requirements.txt
