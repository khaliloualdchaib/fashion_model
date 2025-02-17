FROM python:3.11

WORKDIR /api
RUN pip install pandas flask
COPY . .
ENV PORT=8003
CMD ["python", "app.py"]