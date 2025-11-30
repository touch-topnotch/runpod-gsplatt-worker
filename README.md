# Gaussian Splatting RunPod Worker

Docker образ для обучения 3D Gaussian Splatting моделей на RunPod Serverless.

## Описание

Этот worker обрабатывает видео и создаёт 3D Gaussian Splatting модели:

1. **Скачивает видео** по URL
2. **Извлекает кадры** с помощью ffmpeg
3. **Запускает COLMAP** для определения поз камер
4. **Обучает 3DGS модель** с помощью gaussian-splatting
5. **Загружает результат** на master-server

## Сборка образа

```powershell
# Сборка образа
docker build -t your-registry/runpod-gsplatt-worker:latest .

# Пуш в registry
docker push your-registry/runpod-gsplatt-worker:latest
```

## Переменные окружения

Настройте эти переменные в RunPod Endpoint:

| Переменная | Описание | Обязательно |
|------------|----------|-------------|
| `MASTER_SERVER_URL` | URL вашего master-server (например `https://api.example.com`) | Да |
| `UPLOAD_API_KEY` | API ключ для авторизации загрузки | Нет |

## Деплой на RunPod

1. Зайдите в [RunPod Console](https://www.runpod.io/console/serverless)
2. Создайте новый Serverless Endpoint:
   - **Docker Image**: `your-registry/runpod-gsplatt-worker:latest`
   - **GPU Type**: L4 / A5000 / RTX 3090 (рекомендуется)
   - **Execution Timeout**: 1800000 ms (30 минут)
   - **Env Variables**: настройте переменные выше
3. Сохраните `ENDPOINT_ID`

## API

### Создание задачи

```json
POST /run
{
    "input": {
        "video_url": "https://example.com/video.mp4",
        "scene_id": "my-scene-123",
        "params": {
            "iterations": 30000,
            "fps": 2
        }
    }
}
```

### Ответ

```json
{
    "id": "runpod-job-id",
    "status": "IN_QUEUE"
}
```

### Проверка статуса

```
GET /status/{job_id}
```

### Результат (при успехе)

```json
{
    "status": "COMPLETED",
    "output": {
        "status": "success",
        "scene_id": "my-scene-123",
        "progress": 100,
        "plt_url": "https://your-server.com/files/gsplatt/my-scene-123.zip"
    }
}
```

## Локальное тестирование

```powershell
# Запуск контейнера локально
docker run --gpus all -it \
    -e MASTER_SERVER_URL=https://your-server.com \
    your-registry/runpod-gsplatt-worker:latest

# Тест prepare_from_video.py
docker run --gpus all -it \
    -v /path/to/video.mp4:/workspace/test.mp4 \
    -v /path/to/output:/workspace/output \
    your-registry/runpod-gsplatt-worker:latest \
    python3 prepare_from_video.py --video /workspace/test.mp4 --out /workspace/output --fps 2
```

## Требования к видео

- Формат: MP4, MOV, AVI
- Разрешение: рекомендуется 1080p или ниже
- Длительность: 10-60 секунд оптимально
- Движение камеры: плавное, без резких скачков
- Сцена: статичная (без движущихся объектов)

## Время обработки

Примерное время на GPU L4:
- 20 сек видео, 2 fps, 30k iterations: ~15-20 минут
- 60 сек видео, 2 fps, 30k iterations: ~30-40 минут

## Troubleshooting

### COLMAP не находит достаточно точек

- Увеличьте FPS (больше кадров)
- Убедитесь в хорошем освещении
- Проверьте текстурность сцены

### Out of Memory

- Уменьшите разрешение видео
- Уменьшите количество итераций
- Используйте GPU с большей памятью
