# Запуск проекта

Проект запускается из корня командой:

```bash
uvicorn src.main:app --reload
```

# Зависимости

В `requirements.txt` находятся все зависимости окружения.

# Глобальная обработка ошибок

Сервис возвращает единый формат ошибок:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable message",
  "details": {}
}
```

Используются коды из `ResponseErrorTypes`:

- `VALIDATION_ERROR`
- `INVALID_FEATURE_VECTOR`
- `DATASET_EMPTY`
- `DATA_PREPARATION_ERROR`
- `MODEL_NOT_TRAINED`
- `MODEL_PREDICTION_ERROR`
- `INTERNAL_ERROR`

# Примеры ошибок

## 1) `/model/train` — пустой датасет

Если датасет пустой, вернется:

```json
{
  "code": "DATASET_EMPTY",
  "message": "No data",
  "details": {
    "path": "/model/train"
  }
}
```

## 2) `/predict` — модель не обучена

Если модель отсутствует в памяти/не обучена:

```json
{
  "code": "MODEL_NOT_TRAINED",
  "message": "Model is not trained",
  "details": {
    "path": "/predict"
  }
}
```

## 3) `/predict` — неверные типы или пропущенные поля

Если переданы некорректные значения или не хватает признаков:

```json
{
  "code": "INVALID_FEATURE_VECTOR",
  "message": "Request validation failed",
  "details": {
    "path": "/predict",
    "errors": [
      {
        "loc": ["body", "monthly_fee"],
        "msg": "Input should be a valid number"
      }
    ]
  }
}
```

## 4) `/predict` — лишние поля (неверный набор признаков)

`FeatureVectorChurn` настроен с `extra="forbid"`, поэтому лишние признаки отклоняются:

```json
{
  "code": "INVALID_FEATURE_VECTOR",
  "message": "Request validation failed",
  "details": {
    "path": "/predict",
    "errors": [
      {
        "loc": ["body", "unexpected_feature"],
        "msg": "Extra inputs are not permitted"
      }
    ]
  }
}
```