FROM python:3.11

WORKDIR /app

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure start.sh is executable
RUN chmod +x backend/start.sh

EXPOSE 8000 8501

WORKDIR /app/backend

CMD ["./start.sh"]
