#!/bin/bash

# Переключаемся в директорию проекта
cd "$(dirname "$0")"

# Запускаем database_service
echo "Запуск database_service..."
cd database_sservice
nohup python3 app.py > database.log 2>&1 & 
DB_PID=$!
cd ..

# Запускаем backend_service
echo "Запуск backend_service..."
cd backend_service
nohup python3 app.py > backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Запускаем frontend_service
echo "Запуск frontend_service..."
cd frontend_service
nohup python3 app.py > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "Все сервисы запущены!"
echo "Database PID: $DB_PID"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

# Ждём завершения процессов
wait $DB_PID $BACKEND_PID $FRONTEND_PID

