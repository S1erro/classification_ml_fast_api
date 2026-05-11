# Churn Prediction API

FastAPI-сервис для обучения ML-модели и предсказания оттока клиентов (`churn`).

Сервис читает датасет `data/datasets/churn_dataset.csv`, обучает модель классификации на основе признаков клиента и предоставляет HTTP API для:

- проверки состояния сервиса;
- просмотра информации о датасете;
- обучения модели;
- получения метрик последнего обучения;
- выполнения prediction-запросов.

## Стек

- Python
- FastAPI
- Pydantic
- pandas
- scikit-learn
- joblib
- pytest
- Docker / Docker Compose

## Структура проекта

```text
.
├── data/
│   ├── cached_models/          # сохраненные обученные модели
│   ├── datasets/
│   │   └── churn_dataset.csv   # датасет для обучения
│   └── train_history/          # история обучений
├── docker/
│   ├── Dockerfile
│   ├── compose.yaml
│   └── README.Docker.md
├── src/
│   ├── main.py                 # FastAPI-приложение и endpoints
│   ├── models/                 # Pydantic-модели request/response
│   ├── pipelines/              # сборка ML pipeline
│   ├── types/                  # типы, enum и константы
│   └── utils/                  # утилиты для данных, обучения и сохранения моделей
├── tests/                      # unit и integration tests
├── requirements.txt
└── README.md

Цель сервиса
Цель сервиса — предсказывать вероятность оттока клиента на основе его поведенческих и платежных признаков.

Целевая переменная:

churn = 0 — клиент не ушел;
churn = 1 — клиент ушел.
Сервис поддерживает обучение двух типов моделей:

logreg — Logistic Regression;
rand_forest — Random Forest Classifier.
Формат датасета
Файл датасета находится по пути:


data/datasets/churn_dataset.csv
Ожидаемые колонки:

Колонка	Тип	Описание
monthly_fee	float	ежемесячная плата клиента
usage_hours	float	количество часов использования сервиса
support_requests	int	количество обращений в поддержку
account_age_months	int	возраст аккаунта в месяцах
failed_payments	int	количество неуспешных платежей
region	string	регион клиента
device_type	string	тип устройства
payment_method	string	способ оплаты
autopay_enabled	int	включен ли автоплатеж: 0 или 1
churn	int	целевая переменная: 0 или 1
Пример строки:


monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
Перед обучением сервис:

удаляет строки, где отсутствует churn;
разделяет данные на признаки X и целевую переменную y;
заполняет пропуски в числовых колонках средним значением;
заполняет пропуски в категориальных колонках значением unknown;
делает train/test split со стратификацией.
Локальный запуск
1. Создать виртуальное окружение

python3 -m venv venv
source venv/bin/activate
2. Установить зависимости

pip install -r requirements.txt
3. Запустить сервис

uvicorn src.main:app --reload
После запуска API будет доступно по адресу:


http://127.0.0.1:8000
Swagger UI:


http://127.0.0.1:8000/docs
Запуск в Docker
Из корня проекта:


docker compose -f docker/compose.yaml up --build
После запуска сервис будет доступен по адресу:


http://127.0.0.1:8000
Остановить контейнеры:


docker compose -f docker/compose.yaml down
Основные endpoints
Healthcheck

GET /health
Пример ответа:


{
  "model_available": true,
  "dataset_loaded": true
}
Информация о датасете

GET /dataset/info
Preview датасета

GET /dataset/preview?rows_count=5
Информация о train/test split

GET /dataset/split-info
Статус модели

GET /model/status
JSON schema признаков

GET /model/schema
Метрики последнего обучения

GET /model/metrics
Обучение модели
Endpoint:


POST /model/train
Пример запроса для Logistic Regression:


curl -X POST "http://127.0.0.1:8000/model/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "logreg",
    "hyperparameters": {
      "max_iter": 200
    }
  }'
Пример запроса для Random Forest:


curl -X POST "http://127.0.0.1:8000/model/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "rand_forest",
    "hyperparameters": {
      "n_estimators": 100,
      "random_state": 42
    }
  }'
Пример успешного ответа:


{
  "accuracy": 0.85,
  "f1_score": 0.79,
  "roc_auc": 0.82
}
После обучения модель сохраняется в data/cached_models, а информация об обучении добавляется в data/train_history/train_history.json.

Предсказание churn
Endpoint:


POST /predict
Перед вызовом /predict модель должна быть обучена через /model/train или загружена из сохраненного файла при старте приложения.

Пример запроса:


curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 25.0,
    "usage_hours": 40.0,
    "support_requests": 2,
    "account_age_months": 15,
    "failed_payments": 1,
    "region": "us",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1
  }'
Пример успешного ответа:


{
  "predicted_class": 0,
  "classes_probabilities": [0.72, 0.28]
}
Где:

predicted_class — предсказанный класс churn;
classes_probabilities — вероятности классов [P(0), P(1)].
Формат ошибок
Сервис возвращает ошибки в едином формате:

{
  "code": "ERROR_CODE",
  "message": "Human-readable message",
  "details": {}
}
Пример ошибки, если модель не обучена:

{
  "code": "MODEL_NOT_TRAINED",
  "message": "Model is not trained",
  "details": {
    "path": "/predict"
  }
}
Пример ошибки валидации признаков:

{
  "code": "INVALID_FEATURE_VECTOR",
  "message": "Request validation failed",
  "details": {
    "path": "/predict",
    "errors": []
  }
}
Тесты
Запуск тестов:


python3 -m pytest -q
Текущий результат:


4 passed
Примечания
API-приложение находится в src/main.py.
Pydantic-схемы находятся в src/models/models.py.
ML pipeline собирается в src/pipelines/build_model_pypeline.py.
Подготовка данных находится в src/utils/prepare_data.py.
