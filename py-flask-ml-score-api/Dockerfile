FROM python
WORKDIR /usr/src/app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5050
CMD ["python3", "api.py"]
