# Исследование: Дедупликация сообщений из множества Telegram каналов с использованием LLM

## Обзор проблемы

При агрегации сообщений из нескольких Telegram каналов в один возникает задача дедупликации контента. Простое сравнение хэшей не работает, когда сообщения немного отличаются формулировкой, но несут одинаковый смысл.

**Решение:** Семантическая дедупликация на основе embeddings и LLM.

---

## 🔥 Топ-3 готовых решения на GitHub

### 1. **SemHash** (MinishLab) - РЕКОМЕНДУЕТСЯ
- **GitHub:** https://github.com/MinishLab/semhash
- **Статус:** Активный проект, PyPI пакет
- **Лицензия:** MIT

#### Возможности:
- ⚡ Очень быстрая семантическая дедупликация (120K записей за 5 секунд)
- 🎯 Использует Model2Vec для embeddings + Vicinity для ANN-поиска
- 📊 Поддержка self-deduplication и cross-deduplication
- 🔍 Фильтрация аномалий и поиск репрезентативных образцов

#### Пример использования:
```python
from semhash import SemHash

# Список сообщений из разных каналов
messages = [
    "Биткоин вырос на 10% за день",
    "Bitcoin подорожал на 10% за сутки",
    "Цена BTC увеличилась на 10% за 24 часа",
    "Температура в Москве +5",
]

# Создание индекса
semhash = SemHash.from_records(records=messages)

# Дедупликация
result = semhash.self_deduplicate()
unique_messages = result.selected
print(f"Из {len(messages)} осталось {len(unique_messages)} уникальных")
```

#### Производительность:
- AG News (120K записей): 10.9% дубликатов удалено за 5.2 сек
- WikiText (1.8M записей): 50.9% дубликатов удалено за 83.5 сек

---

### 2. **text-dedup** (ChenghaoMou)
- **GitHub:** https://github.com/ChenghaoMou/text-dedup
- **Статус:** Активный, множество методов
- **Лицензия:** Apache 2.0

#### Возможности:
- 🛠️ Множество алгоритмов: MinHash, SimHash, UniSim (embeddings)
- 📈 Поддержка Spark для TB-scale датасетов
- 🎛️ Гибкая настройка порогов схожести

#### Методы дедупликации:
1. **Exact matching** - точное совпадение (хэши)
2. **Near-deduplication** - MinHash + LSH, SimHash
3. **Semantic** - UniSim/RETSim на основе embeddings

#### Пример использования:
```bash
# MinHash дедупликация
python -m text_dedup.minhash \
  --path "your_dataset" \
  --output "deduplicated_output" \
  --column "text" \
  --threshold 0.8
```

---

### 3. **SemanticDeduplicator** (gkamradt)
- **GitHub:** https://github.com/gkamradt/SemanticDeduplicator
- **Статус:** Alpha, но интересный подход
- **Лицензия:** MIT

#### Возможности:
- 🧠 Гибридный подход: косинусное сходство + LLM-проверка
- 📝 Отлично для управления списками задач/фидбека
- 🎯 Два порога: 0.75 для косинуса, 0.8 для LLM

#### Подход:
1. Первичная фильтрация через косинусное сходство embeddings
2. LLM (OpenAI) принимает финальное решение о дубликатах

#### Пример:
```python
from semantic_deduplicator import SemanticDeduplicator

deduplicator = SemanticDeduplicator(
    context="Telegram channel messages about crypto news"
)

deduplicator.add_item("Bitcoin rose 10% today")
deduplicator.add_item("BTC increased by 10% in 24h")
deduplicator.add_item("Weather forecast for tomorrow")

# Автоматически группирует первые два сообщения
unique_items = deduplicator.get_unique_items()
```

---

## 🔧 Базовый подход с Sentence Transformers

Если нужно простое решение без внешних библиотек:

```python
from sentence_transformers import SentenceTransformer, util

# Загрузка модели (легковесная и быстрая)
model = SentenceTransformer('all-MiniLM-L6-v2')

messages = [
    "Биткоин вырос на 10%",
    "Bitcoin подорожал на 10%",
    "Погода сегодня отличная"
]

# Создание embeddings
embeddings = model.encode(messages, convert_to_tensor=True)

# Вычисление косинусного сходства
cosine_scores = util.cos_sim(embeddings, embeddings)

# Поиск дубликатов (порог 0.85)
threshold = 0.85
duplicates = []

for i in range(len(messages)):
    for j in range(i + 1, len(messages)):
        if cosine_scores[i][j] > threshold:
            duplicates.append((i, j, float(cosine_scores[i][j])))
            print(f"Дубликаты: '{messages[i]}' и '{messages[j]}' (схожесть: {cosine_scores[i][j]:.4f})")
```

---

## 🏗️ Архитектура решения для Telegram

### Компоненты:

```
┌─────────────────────────────────────────────────────────┐
│                    Telegram Channels                     │
│   Channel 1    Channel 2    Channel 3    ...   Channel N │
└────────┬────────────┬────────────┬──────────────┬────────┘
         │            │            │              │
         ▼            ▼            ▼              ▼
┌─────────────────────────────────────────────────────────┐
│              Telethon/Pyrogram Listener                  │
│          (Подписка на события NewMessage)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Message Buffer/Queue                       │
│              (Redis/RabbitMQ/Memory)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Semantic Deduplication Engine                  │
│                                                           │
│  1. Generate embedding (Model2Vec/SentenceTransformer)   │
│  2. ANN search in vector DB (Vicinity/FAISS/Qdrant)     │
│  3. Check similarity threshold (0.85-0.95)               │
│  4. Optional: LLM verification for edge cases            │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴──────────┐
         ▼                      ▼
    ┌─────────┐           ┌──────────┐
    │ Unique  │           │ Duplicate│
    │Messages │           │ Rejected │
    └────┬────┘           └──────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│            Target Telegram Channel                       │
│           (Forward unique messages)                      │
└─────────────────────────────────────────────────────────┘
```

### Код для интеграции с Telethon:

```python
from telethon import TelegramClient, events
from semhash import SemHash
import asyncio

# Инициализация клиента
client = TelegramClient('session', api_id, api_hash)

# Хранилище для дедупликации (можно использовать БД)
message_deduplicator = SemHash.from_records(records=[])
seen_messages = []

# ID исходных каналов и целевого
SOURCE_CHANNELS = [-1001234567890, -1009876543210]  # Замените на реальные
TARGET_CHANNEL = -1001111111111

@client.on(events.NewMessage(chats=SOURCE_CHANNELS))
async def handler(event):
    message_text = event.message.text

    if not message_text:
        return

    # Добавляем сообщение во временное хранилище
    seen_messages.append(message_text)

    # Переиндексируем (в продакшене лучше инкрементально)
    deduplicator = SemHash.from_records(records=seen_messages)

    # Проверяем, уникально ли сообщение
    result = deduplicator.deduplicate(records=[message_text])

    if len(result.selected) > 0:
        # Сообщение уникально - пересылаем
        await client.send_message(TARGET_CHANNEL, message_text)
        print(f"✅ Отправлено: {message_text[:50]}...")
    else:
        # Дубликат - игнорируем
        print(f"❌ Дубликат: {message_text[:50]}...")

client.start()
client.run_until_disconnected()
```

---

## 📊 Сравнение подходов

| Подход | Скорость | Точность | Сложность | Стоимость |
|--------|----------|----------|-----------|-----------|
| Hash (MD5/SHA) | ⚡⚡⚡⚡⚡ | Низкая | Простая | Бесплатно |
| MinHash/SimHash | ⚡⚡⚡⚡ | Средняя | Средняя | Бесплатно |
| **SemHash (embeddings)** | ⚡⚡⚡⚡ | **Высокая** | Средняя | Бесплатно |
| Sentence Transformers | ⚡⚡⚡ | Высокая | Простая | Бесплатно |
| LLM (GPT-4/Claude) | ⚡ | Максимальная | Простая | **Платно** |
| Hybrid (embeddings + LLM) | ⚡⚡ | Максимальная | Средняя | Платно |

---

## 🎯 Рекомендация для вашего проекта

### Оптимальное решение:

**SemHash + Qdrant (или FAISS) + Telethon**

#### Почему:
1. ✅ **Быстрая** обработка потока сообщений в реальном времени
2. ✅ **Точная** семантическая дедупликация (не пропускает синонимы)
3. ✅ **Бесплатная** - работает локально без API вызовов
4. ✅ **Масштабируемая** - поддержка миллионов сообщений
5. ✅ **Готовая** - не нужно обучать модели

#### Для edge cases:
Можно добавить LLM-проверку для сообщений с косинусным сходством 0.75-0.85 (серая зона):

```python
from openai import OpenAI

def llm_verify_duplicate(msg1, msg2, threshold=0.80):
    """Проверка через LLM для неоднозначных случаев"""
    client = OpenAI()

    prompt = f"""Are these two messages semantically identical (same meaning)?

Message 1: {msg1}
Message 2: {msg2}

Answer only: YES or NO"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Дешевая модель
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5
    )

    return response.choices[0].message.content.strip().upper() == "YES"
```

---

## 🚀 План внедрения

### Этап 1: Прототип (1-2 дня)
```bash
pip install semhash telethon
```
- Настроить Telethon для прослушивания 2-3 каналов
- Интегрировать SemHash для базовой дедупликации
- Тестировать на реальных данных

### Этап 2: Оптимизация (3-5 дней)
- Добавить векторную БД (Qdrant/FAISS) для персистентности
- Настроить порог схожести под ваши данные (0.80-0.95)
- Добавить логирование дубликатов для анализа

### Этап 3: Production (неделя)
- Добавить Redis/RabbitMQ для буферизации
- Настроить мониторинг (Prometheus + Grafana)
- Опционально: добавить LLM-проверку для edge cases

---

## 📚 Дополнительные ресурсы

### Альтернативные векторные БД:
- **Qdrant** (рекомендуется) - https://qdrant.tech/
- **FAISS** (от Meta) - https://github.com/facebookresearch/faiss
- **Milvus** - для больших масштабов

### Модели для embeddings:
- `all-MiniLM-L6-v2` - быстрая, легковесная (80MB)
- `all-mpnet-base-v2` - точнее, но тяжелее (420MB)
- `multilingual-e5-large` - для мультиязычного контента

### Useful links:
- SemHash docs: https://github.com/MinishLab/semhash
- Sentence Transformers: https://www.sbert.net/
- Telethon docs: https://docs.telethon.dev/

---

## 💡 Советы по настройке

### Выбор порога схожести:

```python
# Консервативный (мало дубликатов, но точно)
threshold = 0.95

# Сбалансированный (рекомендуется)
threshold = 0.85

# Агрессивный (много дубликатов, возможны ложные срабатывания)
threshold = 0.75
```

### Оптимизация для русского языка:

```python
# Используйте мультиязычные модели
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# или
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
```

---

## ⚠️ Важные замечания

1. **Первый запуск**: при запуске нужно проиндексировать историю сообщений из целевого канала, чтобы не пропустить старые дубликаты
2. **Обновление индекса**: периодически (раз в день/неделю) пересобирать индекс для оптимизации
3. **Медиафайлы**: для изображений/видео используйте perceptual hashing (pHash) + CLIP embeddings
4. **Privacy**: храните только хэши/embeddings, а не сами сообщения (GDPR compliance)

---

## 🎓 Заключение

Для вашей задачи дедупликации Telegram сообщений оптимальным решением будет:

**SemHash + мультиязычная Sentence Transformer модель + Telethon**

Это дает баланс между точностью, скоростью и стоимостью. Если бюджет позволяет, для edge cases можно добавить проверку через дешевую LLM (GPT-4o-mini или Claude Haiku).
