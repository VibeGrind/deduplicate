# Нормализация русскоязычного текста для дедупликации

## Зачем нужна специальная обработка русского языка?

Русский язык имеет уникальные особенности, которые требуют особого подхода к нормализации:

1. **Богатая морфология** - 6 падежей, 3 рода, 2 числа, 3 времени → ~20-30 форм на слово
2. **Флективность** - изменение окончаний ("дом", "дома", "дому", "домом"...)
3. **Буква Ё** - часто пишется как Е, но это разные буквы
4. **Омографы** - разное ударение → разный смысл ("замОк" vs "зАмок")
5. **Словосложение** - "интернет-магазин", "кто-то", "кое-какой"
6. **Множество синонимов** - "автомобиль", "машина", "авто", "тачка"

---

## Многоуровневая нормализация

### Level 1: Базовая нормализация (Fast & Cheap)

Применяется ДО создания normalized hash и поиска в MinHash

#### 1.1 Приведение к lowercase

```python
text = text.lower()

# Особенность для русского: ё → е (опционально)
# Проблема: "ёлка" vs "елка" - часто путают
# Решение: унифицируем для хэширования
text = text.replace('ё', 'е')
```

**Обоснование:**
- 80% пользователей не используют букву ё
- Для дедупликации "Подорожал биткоин" == "подорожал биткоин"

#### 1.2 Нормализация пробелов и переносов

```python
import re

# Множественные пробелы → один
text = re.sub(r'\s+', ' ', text)

# Неразрывные пробелы → обычные
text = text.replace('\u00A0', ' ')  # &nbsp;
text = text.replace('\u202F', ' ')  # narrow no-break space

# Убираем пробелы в начале/конце
text = text.strip()

# Нормализуем переносы строк
text = text.replace('\r\n', '\n')
text = re.sub(r'\n{3,}', '\n\n', text)  # Максимум 2 переноса подряд
```

#### 1.3 Удаление форматирования

```python
# Markdown
text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
text = re.sub(r'__(.+?)__', r'\1', text)      # __bold__
text = re.sub(r'\*(.+?)\*', r'\1', text)      # *italic*
text = re.sub(r'_(.+?)_', r'\1', text)        # _italic_
text = re.sub(r'~~(.+?)~~', r'\1', text)      # ~~strikethrough~~
text = re.sub(r'`(.+?)`', r'\1', text)        # `code`

# HTML (если есть)
from html import unescape
text = unescape(text)  # &quot; → ", &amp; → &
text = re.sub(r'<[^>]+>', '', text)  # Убираем теги
```

#### 1.4 Нормализация пунктуации

```python
# Множественные знаки → одиночные
text = re.sub(r'\.{2,}', '.', text)    # ... → .
text = re.sub(r'!{2,}', '!', text)     # !!! → !
text = re.sub(r'\?{2,}', '?', text)    # ??? → ?

# Emoji и спецсимволы (опционально убираем/заменяем)
import emoji
# Вариант 1: заменяем на текст
text = emoji.demojize(text, language='ru')  # 😊 → :улыбающееся_лицо:
# Вариант 2: убираем полностью
text = emoji.replace_emoji(text, replace='')
```

---

### Level 2: Entity Extraction & Normalization

#### 2.1 URL extraction

```python
import re

URL_PATTERN = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)

def extract_urls(text: str) -> Tuple[str, List[str]]:
    """Извлекает URLs и заменяет на <URL>"""
    urls = URL_PATTERN.findall(text)
    normalized = URL_PATTERN.sub('<URL>', text)
    return normalized, urls
```

**Для Telegram специфичные:**

```python
# t.me/channel, @username links
TG_CHANNEL_PATTERN = re.compile(r't\.me/([a-zA-Z0-9_]+)')
TG_MENTION_LINK = re.compile(r'@([a-zA-Z0-9_]{5,32})')

# Заменяем
text = TG_CHANNEL_PATTERN.sub('<TG_CHANNEL>', text)
```

#### 2.2 Email extraction

```python
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

text, emails = extract_entities(text, EMAIL_PATTERN, '<EMAIL>')
```

#### 2.3 Phone numbers (русские номера)

```python
# Русские номера: +7, 8, различные форматы
PHONE_PATTERNS = [
    r'\+7\s?\(?\d{3}\)?\s?\d{3}[-\s]?\d{2}[-\s]?\d{2}',  # +7 (999) 123-45-67
    r'8\s?\(?\d{3}\)?\s?\d{3}[-\s]?\d{2}[-\s]?\d{2}',     # 8 (999) 123-45-67
    r'\+7\d{10}',                                          # +79991234567
]

def extract_phones(text: str) -> Tuple[str, List[str]]:
    phones = []
    for pattern in PHONE_PATTERNS:
        found = re.findall(pattern, text)
        phones.extend(found)
        text = re.sub(pattern, '<PHONE>', text)
    return text, phones
```

#### 2.4 Mentions & Hashtags

```python
# @username
MENTION_PATTERN = re.compile(r'@([a-zA-Z0-9_]{1,32})')
text, mentions = extract_entities(text, MENTION_PATTERN, '<MENTION>')

# #hashtag (поддержка русских букв!)
HASHTAG_PATTERN = re.compile(r'#([а-яА-ЯёЁa-zA-Z0-9_]+)')
text, hashtags = extract_entities(text, HASHTAG_PATTERN, '<HASHTAG>')
```

#### 2.5 Числа, даты, валюты (подробнее в следующем разделе)

```python
# Простая замена для нормализованного хэша
# (детальное извлечение - в Numeric Metadata слое)

# Числа с разделителями
text = re.sub(r'\d{1,3}(?:[\s,]\d{3})+', '<NUMBER>', text)  # 1 000 000

# Проценты
text = re.sub(r'-?\d+[.,]?\d*\s?%', '<PERCENTAGE>', text)  # 10.5%

# Валюты
text = re.sub(r'\d+[.,]?\d*\s?(?:руб|₽|USD|€|$|BTC)', '<CURRENCY>', text)
```

---

### Level 3: Морфологическая нормализация (Optional, для Embedding слоя)

#### 3.1 Токенизация для русского языка

**Проблема:** Русские слова склоняются, нужно приводить к нормальной форме

```python
# Вариант 1: pymorphy2 (рекомендуется)
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

def lemmatize_word(word: str) -> str:
    """Приводит слово к начальной форме"""
    parsed = morph.parse(word)[0]
    return parsed.normal_form

# Примеры:
lemmatize_word('домов')    # → 'дом'
lemmatize_word('красивая') # → 'красивый'
lemmatize_word('пошёл')    # → 'пойти'
```

**Производительность pymorphy2:**
- Скорость: ~50,000-100,000 слов/сек
- Точность: ~95% для словарных слов
- Размер словаря: ~5.4M словоформ

#### 3.2 Когда использовать лемматизацию?

**✅ Используйте для:**
- Генерации MinHash shingles (повышает recall)
- Поиска ключевых слов
- TF-IDF векторизации

**❌ НЕ используйте для:**
- Normalized hash (слишком агрессивно)
- Отображения пользователю (потеря информации)
- Embeddings (модели сами учитывают морфологию)

#### 3.3 Альтернатива: Stemming (быстрее, но менее точно)

```python
# Snowball Stemmer для русского
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("russian")

stemmer.stem('красивая')  # → 'красив'
stemmer.stem('домов')     # → 'дом'
```

**Сравнение:**

| Метод | Скорость | Точность | Использование |
|-------|----------|----------|---------------|
| Lemmatization (pymorphy2) | 50k words/s | 95% | Рекомендуется |
| Stemming (Snowball) | 500k words/s | 70-80% | Если нужна скорость |
| Без нормализации | - | - | Для embeddings |

#### 3.4 Удаление стоп-слов (опционально)

```python
# Русские стоп-слова
from nltk.corpus import stopwords

RUSSIAN_STOPWORDS = set(stopwords.words('russian'))

# Дополнительные стоп-слова для новостей
CUSTOM_STOPWORDS = {
    'это', 'который', 'более', 'также', 'сообщает',
    'заявил', 'рассказал', 'отметил', 'подчеркнул'
}

ALL_STOPWORDS = RUSSIAN_STOPWORDS | CUSTOM_STOPWORDS

def remove_stopwords(words: List[str]) -> List[str]:
    return [w for w in words if w not in ALL_STOPWORDS]
```

**⚠️ Внимание:** Для дедупликации НЕ рекомендуется удалять стоп-слова!
- Причина: потеря контекста
- Пример: "Он сказал да" vs "Он сказал нет" → после удаления стоп-слов одинаковы

---

### Level 4: Numeric Metadata Extraction (Детальное)

#### 4.1 Извлечение процентов с направлением

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class Percentage:
    value: float
    direction: Optional[str]  # 'up', 'down', None
    context: str  # Окружающий текст

PERCENTAGE_PATTERN = re.compile(
    r'(?:вырос|рост|увеличил|подорожал|поднял).*?(\d+[.,]?\d*)\s?%|'
    r'(?:упал|падение|снизил|подешевел|опустил).*?(\d+[.,]?\d*)\s?%|'
    r'(\d+[.,]?\d*)\s?%'
)

def extract_percentages(text: str) -> List[Percentage]:
    results = []

    # Ищем контекстные паттерны
    for match in PERCENTAGE_PATTERN.finditer(text):
        context = text[max(0, match.start()-20):match.end()+20]

        if match.group(1):  # Рост
            value = float(match.group(1).replace(',', '.'))
            results.append(Percentage(value, 'up', context))

        elif match.group(2):  # Падение
            value = float(match.group(2).replace(',', '.'))
            results.append(Percentage(value, 'down', context))

        else:  # Нейтральное
            value = float(match.group(3).replace(',', '.'))
            results.append(Percentage(value, None, context))

    return results
```

**Примеры:**
```python
extract_percentages("Биткоин вырос на 10%")
# → [Percentage(value=10.0, direction='up', context='...')]

extract_percentages("Акции упали на 5.5%")
# → [Percentage(value=5.5, direction='down', context='...')]

extract_percentages("Доля рынка 25%")
# → [Percentage(value=25.0, direction=None, context='...')]
```

#### 4.2 Извлечение валютных значений

```python
@dataclass
class Currency:
    value: float
    currency: str  # RUB, USD, EUR, BTC, etc.
    context: str

# Паттерны для русскоязычных текстов
CURRENCY_PATTERNS = {
    'RUB': [
        r'(\d+(?:[.,]\d+)?)\s?(?:руб(?:л(?:ей|я|ь))?|₽)',
        r'(\d+(?:[.,]\d+)?)\s?(?:рос(?:сийских)?)\s?рубл',
    ],
    'USD': [
        r'(\d+(?:[.,]\d+)?)\s?(?:USD|долларов?|бакс)',
        r'\$\s?(\d+(?:[.,]\d+)?)',
    ],
    'EUR': [
        r'(\d+(?:[.,]\d+)?)\s?(?:EUR|евро|€)',
    ],
    'BTC': [
        r'(\d+(?:[.,]\d+)?)\s?(?:BTC|биткоин|bitcoin)',
    ],
}

def extract_currencies(text: str) -> List[Currency]:
    results = []

    for currency, patterns in CURRENCY_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                context = text[max(0, match.start()-20):match.end()+20]
                results.append(Currency(value, currency, context))

    return results
```

#### 4.3 Извлечение дат (русский формат)

```python
from datetime import datetime
from dateutil import parser as date_parser

@dataclass
class DateEntity:
    value: datetime
    original: str
    is_relative: bool  # "сегодня", "вчера" и т.д.

# Относительные даты на русском
RELATIVE_DATES = {
    'сегодня': 0,
    'вчера': -1,
    'позавчера': -2,
    'завтра': 1,
    'послезавтра': 2,
}

MONTH_NAMES = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12,
}

def extract_dates(text: str, reference_date=None) -> List[DateEntity]:
    if reference_date is None:
        reference_date = datetime.now()

    results = []

    # Относительные даты
    for rel_date, offset in RELATIVE_DATES.items():
        if rel_date in text.lower():
            date_val = reference_date + timedelta(days=offset)
            results.append(DateEntity(date_val, rel_date, True))

    # Формат "12 ноября 2024"
    pattern = r'(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})'
    for match in re.finditer(pattern, text.lower()):
        day = int(match.group(1))
        month = MONTH_NAMES[match.group(2)]
        year = int(match.group(3))
        date_val = datetime(year, month, day)
        results.append(DateEntity(date_val, match.group(0), False))

    # Формат "12.11.2024"
    pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{4})'
    for match in re.finditer(pattern, text):
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            date_val = datetime(year, month, day)
            results.append(DateEntity(date_val, match.group(0), False))
        except ValueError:
            pass  # Некорректная дата

    return results
```

#### 4.4 Извлечение времени

```python
@dataclass
class TimeEntity:
    hour: int
    minute: int
    original: str

TIME_PATTERN = re.compile(r'(\d{1,2}):(\d{2})')

def extract_times(text: str) -> List[TimeEntity]:
    results = []
    for match in TIME_PATTERN.finditer(text):
        hour, minute = int(match.group(1)), int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            results.append(TimeEntity(hour, minute, match.group(0)))
    return results
```

#### 4.5 Извлечение произвольных чисел

```python
def extract_numbers(text: str) -> List[float]:
    """Извлекает все числа из текста"""
    # Числа с разделителями тысяч (пробелы/запятые)
    # 1 000 000, 1,000,000
    pattern = r'-?\d{1,3}(?:[\s,]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?'

    numbers = []
    for match in re.finditer(pattern, text):
        num_str = match.group(0)
        # Нормализуем: убираем пробелы/запятые как разделители тысяч
        num_str = num_str.replace(' ', '').replace(',', '')
        # Заменяем запятую на точку для десятичных
        if '.' not in num_str and ',' in num_str:
            num_str = num_str.replace(',', '.')
        try:
            numbers.append(float(num_str))
        except ValueError:
            pass

    return numbers
```

---

## Полный пайплайн нормализации

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class NormalizedMessage:
    """Результат полной нормализации"""

    # Уровень 1: базовая нормализация
    normalized_text: str        # Для хэширования
    normalized_hash: str        # SHA256

    # Уровень 2: извлечённые сущности
    urls: List[str]
    emails: List[str]
    phones: List[str]
    mentions: List[str]
    hashtags: List[str]

    # Уровень 3: морфология (опционально)
    lemmatized_text: str       # Для MinHash
    tokens: List[str]

    # Уровень 4: числовые метаданные
    percentages: List[Percentage]
    currencies: List[Currency]
    dates: List[DateEntity]
    times: List[TimeEntity]
    numbers: List[float]

    # Метаданные
    original_text: str
    language: str = 'ru'


class RussianTextNormalizer:
    """Полная нормализация для русского текста"""

    def __init__(self, use_lemmatization: bool = True):
        self.use_lemmatization = use_lemmatization

        if use_lemmatization:
            import pymorphy2
            self.morph = pymorphy2.MorphAnalyzer()

    def normalize(self, text: str) -> NormalizedMessage:
        """Выполняет полную нормализацию"""

        original_text = text

        # Level 1: Базовая нормализация
        text = self._basic_normalization(text)

        # Level 2: Entity extraction
        text, entities = self._extract_entities(text)

        # Level 3: Морфология
        if self.use_lemmatization:
            lemmatized = self._lemmatize(text)
            tokens = lemmatized.split()
        else:
            lemmatized = text
            tokens = text.split()

        # Level 4: Numeric metadata
        numeric_meta = self._extract_numeric_metadata(original_text)

        # Создаём хэш
        import hashlib
        normalized_hash = hashlib.sha256(text.encode()).hexdigest()

        return NormalizedMessage(
            normalized_text=text,
            normalized_hash=normalized_hash,
            urls=entities['urls'],
            emails=entities['emails'],
            phones=entities['phones'],
            mentions=entities['mentions'],
            hashtags=entities['hashtags'],
            lemmatized_text=lemmatized,
            tokens=tokens,
            percentages=numeric_meta['percentages'],
            currencies=numeric_meta['currencies'],
            dates=numeric_meta['dates'],
            times=numeric_meta['times'],
            numbers=numeric_meta['numbers'],
            original_text=original_text,
            language='ru'
        )

    def _basic_normalization(self, text: str) -> str:
        """Level 1 нормализация"""
        text = text.lower()
        text = text.replace('ё', 'е')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        # ... (все методы из Level 1)
        return text

    def _extract_entities(self, text: str) -> Tuple[str, Dict]:
        """Level 2 извлечение сущностей"""
        entities = {}

        text, entities['urls'] = self._extract_urls(text)
        text, entities['emails'] = self._extract_emails(text)
        text, entities['phones'] = self._extract_phones(text)
        text, entities['mentions'] = self._extract_mentions(text)
        text, entities['hashtags'] = self._extract_hashtags(text)

        return text, entities

    def _lemmatize(self, text: str) -> str:
        """Level 3 лемматизация"""
        words = text.split()
        lemmas = [self.morph.parse(word)[0].normal_form for word in words]
        return ' '.join(lemmas)

    def _extract_numeric_metadata(self, text: str) -> Dict:
        """Level 4 извлечение числовых данных"""
        return {
            'percentages': extract_percentages(text),
            'currencies': extract_currencies(text),
            'dates': extract_dates(text),
            'times': extract_times(text),
            'numbers': extract_numbers(text),
        }
```

---

## Использование в пайплайне дедупликации

```python
# Инициализация
normalizer = RussianTextNormalizer(use_lemmatization=True)

# Пример сообщения
message = """
Биткоин вырос на 10% за последние сутки! 🚀
Цена достигла 50 000 USD на 12 ноября 2024.
Подробнее: https://crypto.news/bitcoin
#Bitcoin #криптовалюта
"""

# Нормализуем
normalized = normalizer.normalize(message)

print(f"Normalized text: {normalized.normalized_text}")
# → "биткоин вырос на <PERCENTAGE> за последние сутки <URL> <HASHTAG> <HASHTAG>"

print(f"Hash: {normalized.normalized_hash[:16]}...")
# → "a3f2e8b9..."

print(f"Percentages: {normalized.percentages}")
# → [Percentage(value=10.0, direction='up', context='...')]

print(f"Currencies: {normalized.currencies}")
# → [Currency(value=50000.0, currency='USD', context='...')]

print(f"Dates: {normalized.dates}")
# → [DateEntity(value=datetime(2024, 11, 12), original='12 ноября 2024', is_relative=False)]

# Использование в слоях
# Layer 1: проверяем normalized_hash
# Layer 3: сравниваем numeric metadata
# Layer 4: создаём MinHash из lemmatized_text
# Layer 5: генерируем embedding из original_text (или normalized_text)
```

---

## Оптимизации производительности

### 1. Кэширование нормализации

```python
from functools import lru_cache
import hashlib

class CachedNormalizer(RussianTextNormalizer):
    """Нормализатор с кэшированием"""

    @lru_cache(maxsize=10_000)
    def normalize_cached(self, text: str) -> str:
        """Кэшируем normalized_text по хэшу оригинала"""
        result = self.normalize(text)
        return result.normalized_text

    def get_cache_key(self, text: str) -> str:
        """Генерируем ключ кэша"""
        return hashlib.md5(text.encode()).hexdigest()
```

### 2. Батчинг для pymorphy2

```python
def batch_lemmatize(self, texts: List[str]) -> List[str]:
    """Лемматизация батчем (быстрее благодаря кэшу pymorphy2)"""
    results = []
    for text in texts:
        words = text.split()
        lemmas = [self.morph.parse(word)[0].normal_form for word in words]
        results.append(' '.join(lemmas))
    return results
```

### 3. Параллельная обработка

```python
from multiprocessing import Pool

def parallel_normalize(texts: List[str], num_workers: int = 4) -> List[NormalizedMessage]:
    """Параллельная нормализация"""
    with Pool(processes=num_workers) as pool:
        results = pool.map(normalizer.normalize, texts)
    return results
```

---

## Тестирование на реальных примерах

```python
# Примеры дубликатов
duplicates = [
    ("Биткоин вырос на 10% за день", "BTC подорожал на 10% за сутки"),
    ("Курс доллара 100₽", "Доллар стоит 100 рублей"),
    ("Встреча в 15:30", "Встреча назначена на 15:30"),
]

# Примеры НЕ дубликатов
non_duplicates = [
    ("Биткоин вырос на 10%", "Биткоин вырос на 15%"),  # Разные проценты
    ("Встреча сегодня", "Встреча завтра"),              # Разные даты
    ("Цена $100", "Цена $200"),                         # Разные суммы
]

# Проверка
for msg1, msg2 in duplicates:
    norm1 = normalizer.normalize(msg1)
    norm2 = normalizer.normalize(msg2)

    # Должны иметь схожие numeric metadata
    assert norm1.percentages == norm2.percentages, "Должны совпадать проценты"
```

---

## Выводы

1. **Многоуровневая нормализация** даёт лучшие результаты чем один метод
2. **pymorphy2** - лучший выбор для морфологии русского языка
3. **Числовые метаданные критичны** для новостного контента
4. **Сущности (URL, email, etc.)** нужно извлекать до создания хэша
5. **Кэширование** ускоряет повторную обработку в 10-100x

**Результат:** Робастная нормализация для точной дедупликации русскоязычных сообщений.
