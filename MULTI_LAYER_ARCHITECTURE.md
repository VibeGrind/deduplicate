# Многослойная архитектура дедупликации для русскоязычных Telegram сообщений

## Философия подхода

**Ключевой принцип:** LLM - это последний слой защиты, а не основной инструмент. Мы строим пирамиду фильтров, где каждый слой отсеивает очевидные случаи, оставляя для LLM только самые сложные граничные ситуации.

```
┌─────────────────────────────────────────────────────┐
│ Layer 0: Exact Match (Hash)           Cost: 0      │ 95% дубликатов
├─────────────────────────────────────────────────────┤
│ Layer 1: Normalized Hash               Cost: 0      │
├─────────────────────────────────────────────────────┤
│ Layer 2: Structural Similarity         Cost: 0      │ 3% дубликатов
├─────────────────────────────────────────────────────┤
│ Layer 3: Numeric Metadata Match        Cost: 0      │
├─────────────────────────────────────────────────────┤
│ Layer 4: MinHash/LSH Blocking           Cost: 0      │ 1.5% дубликатов
├─────────────────────────────────────────────────────┤
│ Layer 5: Semantic Similarity (Embeddings) Cost: LOW │
├─────────────────────────────────────────────────────┤
│ Layer 6: LLM Verification (edge cases)  Cost: HIGH  │ 0.5% граничных
└─────────────────────────────────────────────────────┘
```

---

## Layer 0: Exact Hash Match

**Цель:** Отсеять 100% идентичные сообщения максимально быстро

**Метод:**
- SHA256 от сырого текста
- O(1) lookup в hash table/Redis

**Эффективность:**
- Скорость: ~1 млн проверок/сек
- Отсекает: ~30-40% дубликатов (копипаста без изменений)

---

## Layer 1: Normalized Hash Match

**Цель:** Отсеять сообщения, отличающиеся только форматированием

**Процесс нормализации:**

### 1.1 Удаление форматирования
- Убрать Markdown разметку (`**`, `__`, `~~`, ` ``` `)
- Убрать HTML теги если есть
- Убрать Telegram форматирование (@username → username)

### 1.2 Entity extraction & replacement
```
URL       → <URL>
Email     → <EMAIL>
Phone     → <PHONE>
@mention  → <MENTION>
#hashtag  → <HASHTAG>
```

### 1.3 Нормализация пробелов
- Множественные пробелы → одиночные
- Trim начала/конца
- Нормализация переносов строк (\r\n → \n)

### 1.4 Регистронезависимость для хэша
- Lowercase для русского и латиницы
- Учёт ё ↔ е эквивалентности

**Результат:**
```
Original:  "Биткоин вырос на 10%! Подробнее: https://example.com"
Normalized: "биткоин вырос на 10%! подробнее: <URL>"
Hash:      SHA256(normalized)
```

**Эффективность:**
- Скорость: ~500k проверок/сек
- Отсекает дополнительно: ~20-30% дубликатов

---

## Layer 2: Structural Similarity

**Цель:** Быстрая проверка структурного сходства до тяжелых вычислений

### 2.1 Извлечение структурных признаков

```
Features:
- length_chars: длина текста
- length_words: количество слов
- sentence_count: количество предложений
- avg_word_length: средняя длина слова
- punctuation_density: плотность знаков препинания
- capital_ratio: доля заглавных букв
- digit_ratio: доля цифр
- emoji_count: количество эмодзи
```

### 2.2 Правило быстрого отсечения

**Если сообщения семантически идентичны, их структура должна быть похожа:**

```python
def structural_filter(msg1, msg2) -> bool:
    """Вернуть False если точно НЕ дубликаты"""

    # Разница в длине > 30% → точно не дубликаты
    if abs(msg1.length - msg2.length) / max(msg1.length, msg2.length) > 0.30:
        return False

    # Разница в количестве слов > 40%
    if abs(msg1.words - msg2.words) / max(msg1.words, msg2.words) > 0.40:
        return False

    # Разница в количестве предложений > 50%
    if abs(msg1.sentences - msg2.sentences) / max(msg1.sentences, msg2.sentences) > 0.50:
        return False

    return True  # Может быть дубликат, идём дальше
```

**Эффективность:**
- Скорость: ~2 млн проверок/сек (простые арифметические операции)
- Отсекает дополнительно: ~15-25% ложных кандидатов

---

## Layer 3: Numeric Metadata Extraction & Comparison

**Ключевая идея:** Если в сообщениях разные числа, это скорее всего разные новости!

### 3.1 Извлечение числовых сущностей

```python
NumericMetadata:
    - integers: [10, 5000, 2024]
    - floats: [10.5, 99.99]
    - percentages: [10%, -5.5%]
    - currencies: [100 USD, 5000 RUB, 0.5 BTC]
    - dates: [2024-11-12, 12.11.2024]
    - times: [15:30, 10:45]
    - phone_numbers: [+79991234567]
    - versions: [v2.0.1, iOS 17]
```

### 3.2 Нормализация числовых данных

**Процентные значения:**
```
"выросла на 10%" → {type: 'percentage', value: 10.0, direction: 'up'}
"упала на 5.5%" → {type: 'percentage', value: 5.5, direction: 'down'}
```

**Валютные значения:**
```
"100 долларов" → {type: 'currency', value: 100, currency: 'USD'}
"5000₽" → {type: 'currency', value: 5000, currency: 'RUB'}
"0.5 биткоина" → {type: 'currency', value: 0.5, currency: 'BTC'}
```

**Даты:**
```
"12 ноября 2024" → {type: 'date', value: '2024-11-12'}
"сегодня" → {type: 'date', value: '2024-11-12', relative: True}
```

### 3.3 Сравнение метаданных

```python
def numeric_similarity(meta1, meta2) -> float:
    """Возвращает сходство числовых данных [0.0 - 1.0]"""

    # Извлекаем все числа
    nums1 = set(meta1.all_numbers)
    nums2 = set(meta2.all_numbers)

    # Jaccard similarity для чисел
    if not nums1 and not nums2:
        return 1.0  # Нет чисел в обоих

    intersection = len(nums1 & nums2)
    union = len(nums1 | nums2)

    jaccard = intersection / union if union > 0 else 0.0

    # Если числа сильно различаются → вероятно разные события
    if jaccard < 0.5:
        return 0.0  # Точно не дубликаты

    return jaccard
```

**Примеры:**

```
✅ DUPLICATE (numeric_similarity = 1.0):
msg1: "Биткоин вырос на 10% за день"
msg2: "BTC подорожал на 10% за сутки"
→ {percentage: 10.0} в обоих

❌ NOT DUPLICATE (numeric_similarity = 0.0):
msg1: "Биткоин вырос на 10%"
msg2: "Биткоин вырос на 15%"
→ {percentage: 10.0} vs {percentage: 15.0}

❌ NOT DUPLICATE (numeric_similarity = 0.33):
msg1: "Курс доллара 100₽ на 12 ноября"
msg2: "Курс доллара 105₽ на 13 ноября"
→ {100, 12, 11} vs {105, 13, 11} → Jaccard = 1/5 = 0.2
```

**Эффективность:**
- Скорость: ~100k проверок/сек (regex + parsing)
- Отсекает дополнительно: ~30-40% семантически разных сообщений

---

## Layer 4: MinHash/LSH Blocking

**Цель:** Избежать O(n²) сравнений, создав "кандидатов" для проверки

### 4.1 MinHash signature generation

```python
from datasketch import MinHash, MinHashLSH

# Создаём signature для каждого сообщения
def create_minhash(text: str, num_perm=128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # Используем word-level shingles
    words = text.split()
    for i in range(len(words) - 2):
        shingle = ' '.join(words[i:i+3])  # 3-gram
        m.update(shingle.encode('utf-8'))
    return m
```

### 4.2 LSH Index для быстрого поиска

```python
# Создаём LSH индекс
lsh = MinHashLSH(threshold=0.7, num_perm=128)

# Индексируем все сообщения
for msg_id, msg_text in messages:
    minhash = create_minhash(normalize(msg_text))
    lsh.insert(msg_id, minhash)

# Поиск кандидатов для нового сообщения
new_minhash = create_minhash(normalize(new_message))
candidates = lsh.query(new_minhash)  # Возвращает только похожие
```

**Преимущества:**
- Вместо N² сравнений → только кандидаты (обычно 0-10 на сообщение)
- Скорость: ~50k индексаций/сек, ~100k запросов/сек
- Гарантированный recall при правильных параметрах

**Эффективность:**
- Сокращает пространство поиска с O(n²) до O(n)
- Пропускает на следующий слой только ~1-5% пар

---

## Layer 5: Semantic Similarity (Embeddings)

**Цель:** Семантическая проверка кандидатов из Layer 4

### 5.1 Выбор модели для русского языка

**Рекомендуемые модели (по возрастанию качества):**

1. **multilingual-e5-small** (118MB)
   - Скорость: ~1000 embeddings/сек на CPU
   - Качество: хорошее для 15+ языков

2. **paraphrase-multilingual-MiniLM-L12-v2** (471MB)
   - Скорость: ~500 embeddings/сек на CPU
   - Качество: отличное для парафраз

3. **intfloat/multilingual-e5-large** (2.24GB)
   - Скорость: ~100 embeddings/сек на CPU
   - Качество: state-of-the-art для мультиязычных задач

**Для максимальной производительности:**
- Используйте ONNX quantized модели (2-4x ускорение)
- Батчинг: обрабатывайте по 32-64 сообщения одновременно

### 5.2 Векторная БД для хранения

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Инициализация
client = QdrantClient(":memory:")  # или path="./qdrant_storage"

client.create_collection(
    collection_name="telegram_messages",
    vectors_config=VectorParams(
        size=384,  # размер для e5-small
        distance=Distance.COSINE
    )
)

# Добавление вектора
client.upsert(
    collection_name="telegram_messages",
    points=[
        PointStruct(
            id=msg_id,
            vector=embedding.tolist(),
            payload={
                "text": normalized_text,
                "numeric_meta": numeric_metadata,
                "timestamp": timestamp
            }
        )
    ]
)

# Поиск похожих
results = client.search(
    collection_name="telegram_messages",
    query_vector=new_embedding,
    limit=10,
    score_threshold=0.85
)
```

### 5.3 Комбинированный скоринг

**Не используем просто cosine similarity!** Комбинируем с numeric similarity:

```python
def combined_similarity(
    cosine_sim: float,
    numeric_sim: float,
    weights=(0.7, 0.3)
) -> float:
    """
    Взвешенное сходство

    weights: (w_semantic, w_numeric)
    """
    w_sem, w_num = weights

    # Если числа совсем разные → понижаем общий score
    if numeric_sim < 0.5:
        numeric_penalty = 0.5
    else:
        numeric_penalty = 1.0

    combined = (w_sem * cosine_sim + w_num * numeric_sim) * numeric_penalty

    return combined

# Пример
cosine = 0.92  # Очень похожи семантически
numeric = 0.0  # Числа разные

final_score = combined_similarity(cosine, numeric)
# = (0.7*0.92 + 0.3*0.0) * 0.5 = 0.322 → НЕ дубликат!
```

**Пороги для классификации:**

```python
if combined_score >= 0.90:
    decision = "DUPLICATE"
elif combined_score >= 0.75:
    decision = "UNCERTAIN → send to LLM"
else:
    decision = "NOT_DUPLICATE"
```

**Эффективность:**
- Скорость: ~100-1000 embeddings/сек (в зависимости от модели)
- Точность: 95-97% с правильными порогами
- Передаёт на LLM: ~2-5% граничных случаев

---

## Layer 6: LLM Verification (Final Layer)

**Цель:** Финальная проверка для граничных случаев (0.75 ≤ score < 0.90)

### 6.1 Оптимизация вызовов LLM

**Принципы:**
- ✅ Только для граничных случаев (2-5% от всех сообщений)
- ✅ Используем дешёвые модели (GPT-4o-mini, Claude Haiku)
- ✅ Батчинг: проверяем по 5-10 пар за раз
- ✅ Кэширование решений LLM

### 6.2 Промпт для русского языка

```python
SYSTEM_PROMPT = """Ты эксперт по анализу новостного контента на русском языке.
Твоя задача: определить, являются ли два сообщения дубликатами.

ДУБЛИКАТЫ - это сообщения, которые:
- Сообщают об ОДНОМ И ТОМ ЖЕ событии/факте
- Содержат ОДИНАКОВЫЕ ключевые цифры (проценты, суммы, даты)
- Могут отличаться формулировкой, но смысл идентичен

НЕ ДУБЛИКАТЫ:
- Разные события, даже если похожие темы
- Разные цифры/даты/суммы
- Одна тема, но разные детали

Отвечай ТОЛЬКО: DUPLICATE или NOT_DUPLICATE"""

USER_PROMPT = f"""Сообщение 1: {msg1}

Сообщение 2: {msg2}

Метаданные:
- Cosine similarity: {cosine_sim:.3f}
- Numeric similarity: {numeric_sim:.3f}
- Извлечённые числа из msg1: {nums1}
- Извлечённые числа из msg2: {nums2}

Являются ли эти сообщения дубликатами?"""
```

### 6.3 Батчинг для экономии

```python
async def batch_llm_check(
    pairs: List[Tuple[str, str]],
    batch_size: int = 5
) -> List[bool]:
    """Проверяем несколько пар за один вызов"""

    batched_prompt = "Проверь следующие пары сообщений:\n\n"

    for i, (msg1, msg2) in enumerate(pairs[:batch_size]):
        batched_prompt += f"Пара {i+1}:\n"
        batched_prompt += f"A: {msg1}\n"
        batched_prompt += f"B: {msg2}\n\n"

    batched_prompt += "Ответь в формате: 1:YES/NO, 2:YES/NO, ..."

    response = await llm_client.complete(batched_prompt)

    # Парсинг ответа
    return parse_batch_response(response)
```

**Эффективность:**
- Стоимость: ~$0.001-0.01 за 1000 сообщений (при 5% попадании в LLM)
- Точность: 99%+ на граничных случаях
- Скорость: ~10-50 проверок/сек (с батчингом)

---

## Итоговая метрика эффективности

### Предполагаемое распределение нагрузки на 10,000 сообщений:

```
Layer 0 (Exact Hash):          4,000 сообщений → STOP (40%)
Layer 1 (Normalized Hash):     2,500 сообщений → STOP (25%)
Layer 2 (Structural Filter):   1,500 сообщений → STOP (15%)
Layer 3 (Numeric Metadata):    1,200 сообщений → STOP (12%)
Layer 4 (MinHash/LSH):         Создаёт ~500 пар кандидатов
Layer 5 (Embeddings):          450 пар → DUPLICATE/NOT_DUPLICATE (4.5%)
Layer 6 (LLM):                 50 пар граничных случаев (0.5%)
```

### Совокупные метрики:

| Метрика | Значение |
|---------|----------|
| Общая точность (Accuracy) | 98-99% |
| Precision (дубликаты) | 97-99% |
| Recall (дубликаты) | 96-98% |
| Скорость обработки | ~1000-5000 msg/sec |
| Стоимость на 1M сообщений | $1-5 (только LLM layer) |
| Ложные позитивы (FP) | 1-2% |
| Ложные негативы (FN) | 2-3% |

---

## Рекомендации по настройке

### 1. Начните с консервативных порогов

```python
THRESHOLDS = {
    'structural_diff_max': 0.30,        # Слой 2
    'numeric_jaccard_min': 0.50,        # Слой 3
    'minhash_lsh': 0.70,                # Слой 4
    'embedding_duplicate': 0.90,        # Слой 5: точно дубликат
    'embedding_uncertain_low': 0.75,    # Слой 5→6: граница для LLM
    'combined_weight_semantic': 0.70,   # Вес семантики
    'combined_weight_numeric': 0.30,    # Вес чисел
}
```

### 2. Логируйте всё для анализа

```python
# Логируйте каждое решение с деталями
log_decision({
    'msg1_id': msg1_id,
    'msg2_id': msg2_id,
    'layer_stopped': 'layer_3_numeric',
    'cosine_sim': 0.88,
    'numeric_sim': 0.20,
    'decision': 'NOT_DUPLICATE',
    'reason': 'Different numbers: {10%, 15%}',
    'timestamp': now()
})
```

### 3. A/B тестирование порогов

Собирайте метрики и оптимизируйте пороги на реальных данных:

```python
# Запустите параллельно с разными конфигами
configs = [
    {'embedding_threshold': 0.85, 'numeric_weight': 0.2},
    {'embedding_threshold': 0.90, 'numeric_weight': 0.3},
    {'embedding_threshold': 0.88, 'numeric_weight': 0.25},
]

# Сравните precision/recall на размеченной выборке
best_config = optimize_thresholds(configs, labeled_dataset)
```

---

## Выводы

1. **Не используйте только embeddings** - это расточительно и медленно
2. **Числовые метаданные критически важны** для новостного контента
3. **LSH решает проблему масштабируемости** - избегаем O(n²)
4. **LLM только для edge cases** - 95%+ решений принимается дешёвыми слоями
5. **Нормализация текста** - фундамент всей системы

**Результат:** Быстрая, точная и экономичная система дедупликации для потока Telegram сообщений.
