"""
Тестирование методов дедупликации
Сравнение разных подходов без подключения к Telegram

Использование:
    python test_deduplication.py
"""

import time
from typing import List, Tuple
from dataclasses import dataclass

# Проверяем доступность библиотек
try:
    from semhash import SemHash
    SEMHASH_AVAILABLE = True
except ImportError:
    SEMHASH_AVAILABLE = False
    print("⚠️ SemHash не установлен: pip install semhash")

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Sentence-Transformers не установлен: pip install sentence-transformers")

import hashlib


@dataclass
class TestResult:
    """Результат теста"""
    method: str
    duplicates_found: int
    execution_time: float
    accuracy: float


# Тестовые данные - реалистичные примеры сообщений
TEST_MESSAGES = [
    # Группа 1: Криптоновости (дубликаты)
    "Bitcoin вырос на 10% за последние 24 часа",
    "Биткоин подорожал на 10% за сутки",
    "BTC увеличился на 10 процентов за день",

    # Группа 2: Погода (дубликаты)
    "Завтра в Москве ожидается дождь",
    "Прогноз на завтра: в столице будет дождливо",
    "Синоптики обещают дождь в Москве завтра",

    # Группа 3: Уникальные сообщения
    "Новый iPhone представят в сентябре",
    "Доллар упал до 90 рублей",
    "В парламенте обсуждают новый закон",

    # Группа 4: Похожие, но не дубликаты
    "Температура в Москве +15",
    "Температура в Санкт-Петербурге +15",

    # Группа 5: Технические новости (дубликаты)
    "OpenAI выпустила новую версию GPT-5",
    "Компания OpenAI анонсировала GPT-5",
    "GPT-5 от OpenAI уже доступна",

    # Группа 6: Ещё уникальные
    "Цены на нефть растут третий день подряд",
    "Сегодня состоится матч Россия - Бразилия",
    "Учёные открыли новую планету",
]

# Ожидаемые дубликаты (для проверки точности)
EXPECTED_DUPLICATES = [
    # Группа 1: индексы 0, 1, 2
    (0, 1), (0, 2), (1, 2),
    # Группа 2: индексы 3, 4, 5
    (3, 4), (3, 5), (4, 5),
    # Группа 5: индексы 11, 12, 13
    (11, 12), (11, 13), (12, 13),
]


def test_hash_deduplication(messages: List[str], threshold: float = 1.0) -> TestResult:
    """Тест: простое хеширование (baseline)"""
    start_time = time.time()

    seen_hashes = set()
    duplicates_count = 0

    for msg in messages:
        msg_hash = hashlib.md5(msg.encode()).hexdigest()

        if msg_hash in seen_hashes:
            duplicates_count += 1
        else:
            seen_hashes.add(msg_hash)

    execution_time = time.time() - start_time

    # Hash не найдёт семантические дубликаты, так что accuracy будет низкая
    accuracy = 0.0

    return TestResult(
        method="MD5 Hash",
        duplicates_found=duplicates_count,
        execution_time=execution_time,
        accuracy=accuracy
    )


def test_semhash_deduplication(messages: List[str], threshold: float = 0.85) -> TestResult:
    """Тест: SemHash"""
    if not SEMHASH_AVAILABLE:
        return None

    start_time = time.time()

    # Создание индекса и дедупликация
    semhash = SemHash.from_records(records=messages, threshold=threshold)
    result = semhash.self_deduplicate()

    duplicates_count = len(messages) - len(result.selected)
    execution_time = time.time() - start_time

    # Оценка точности (упрощённая)
    accuracy = calculate_accuracy(messages, result.selected, EXPECTED_DUPLICATES)

    return TestResult(
        method=f"SemHash (threshold={threshold})",
        duplicates_found=duplicates_count,
        execution_time=execution_time,
        accuracy=accuracy
    )


def test_sentence_transformers(
    messages: List[str],
    threshold: float = 0.85,
    model_name: str = 'all-MiniLM-L6-v2'
) -> TestResult:
    """Тест: Sentence Transformers"""
    if not TRANSFORMERS_AVAILABLE:
        return None

    start_time = time.time()

    # Загрузка модели
    model = SentenceTransformer(model_name)

    # Генерация embeddings
    embeddings = model.encode(messages, convert_to_tensor=True)

    # Вычисление косинусного сходства
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # Поиск дубликатов
    duplicates_count = 0
    found_duplicates = []

    for i in range(len(messages)):
        for j in range(i + 1, len(messages)):
            similarity = cosine_scores[i][j].item()

            if similarity >= threshold:
                duplicates_count += 1
                found_duplicates.append((i, j))

    execution_time = time.time() - start_time

    # Оценка точности
    accuracy = calculate_accuracy_pairs(found_duplicates, EXPECTED_DUPLICATES)

    return TestResult(
        method=f"Sentence-Transformers ({model_name}, threshold={threshold})",
        duplicates_found=duplicates_count,
        execution_time=execution_time,
        accuracy=accuracy
    )


def calculate_accuracy(original: List[str], deduplicated: List[str], expected_dups: List[Tuple]) -> float:
    """Упрощённая оценка точности (для SemHash)"""
    # Для SemHash мы получаем список уникальных сообщений
    # Точная оценка сложна без информации о том, какие именно были удалены
    # Используем эвристику: ожидаем удалить 9 дубликатов (из 18 сообщений)

    expected_unique = len(original) - len(expected_dups)
    actual_unique = len(deduplicated)

    # Чем ближе к ожидаемому количеству, тем выше accuracy
    accuracy = 1.0 - abs(expected_unique - actual_unique) / len(original)

    return max(0.0, min(1.0, accuracy))


def calculate_accuracy_pairs(found: List[Tuple], expected: List[Tuple]) -> float:
    """Точная оценка точности (для попарного сравнения)"""
    if not expected:
        return 1.0

    found_set = set(found)
    expected_set = set(expected)

    # True Positives: правильно найденные дубликаты
    tp = len(found_set & expected_set)

    # False Positives: ложные срабатывания
    fp = len(found_set - expected_set)

    # False Negatives: пропущенные дубликаты
    fn = len(expected_set - found_set)

    # Precision и Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1-score как мера accuracy
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def print_results(results: List[TestResult]):
    """Красивый вывод результатов"""
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МЕТОДОВ ДЕДУПЛИКАЦИИ")
    print("=" * 80)
    print(f"\n📝 Всего сообщений: {len(TEST_MESSAGES)}")
    print(f"🎯 Ожидаемых дубликатов: {len(EXPECTED_DUPLICATES)}\n")

    print(f"{'Метод':<50} {'Дубликаты':<12} {'Время (с)':<12} {'Точность':<10}")
    print("-" * 80)

    for result in results:
        if result:
            accuracy_str = f"{result.accuracy:.2%}" if result.accuracy > 0 else "N/A"
            print(f"{result.method:<50} {result.duplicates_found:<12} {result.execution_time:<12.4f} {accuracy_str:<10}")

    print("=" * 80)


def print_detailed_examples():
    """Показать примеры дубликатов"""
    print("\n" + "=" * 80)
    print("📋 ПРИМЕРЫ ОЖИДАЕМЫХ ДУБЛИКАТОВ")
    print("=" * 80)

    duplicate_groups = {
        "Криптоновости": [0, 1, 2],
        "Погода": [3, 4, 5],
        "OpenAI/GPT-5": [11, 12, 13]
    }

    for group_name, indices in duplicate_groups.items():
        print(f"\n🔹 {group_name}:")
        for idx in indices:
            print(f"   [{idx}] {TEST_MESSAGES[idx]}")


def main():
    """Главная функция"""
    print("🧪 Запуск тестов дедупликации...\n")

    results = []

    # Тест 1: Baseline (Hash)
    print("1️⃣ Тестирование MD5 Hash...")
    results.append(test_hash_deduplication(TEST_MESSAGES))

    # Тест 2: SemHash
    if SEMHASH_AVAILABLE:
        print("2️⃣ Тестирование SemHash...")
        for threshold in [0.75, 0.85, 0.95]:
            results.append(test_semhash_deduplication(TEST_MESSAGES, threshold=threshold))
    else:
        print("⏭️ SemHash пропущен (не установлен)")

    # Тест 3: Sentence Transformers
    if TRANSFORMERS_AVAILABLE:
        print("3️⃣ Тестирование Sentence Transformers...")

        # Быстрая модель
        results.append(test_sentence_transformers(
            TEST_MESSAGES,
            threshold=0.85,
            model_name='all-MiniLM-L6-v2'
        ))

        # Мультиязычная модель (если хватает памяти)
        try:
            results.append(test_sentence_transformers(
                TEST_MESSAGES,
                threshold=0.85,
                model_name='paraphrase-multilingual-mpnet-base-v2'
            ))
        except Exception as e:
            print(f"   ⚠️ Ошибка с мультиязычной моделью: {e}")

    else:
        print("⏭️ Sentence Transformers пропущен (не установлен)")

    # Вывод результатов
    print_results(results)

    # Детальные примеры
    print_detailed_examples()

    # Рекомендации
    print("\n" + "=" * 80)
    print("💡 РЕКОМЕНДАЦИИ")
    print("=" * 80)

    best_result = max([r for r in results if r and r.accuracy > 0], key=lambda x: x.accuracy, default=None)

    if best_result:
        print(f"\n🏆 Лучший результат: {best_result.method}")
        print(f"   - Точность: {best_result.accuracy:.2%}")
        print(f"   - Скорость: {best_result.execution_time:.4f} сек")
        print(f"   - Найдено дубликатов: {best_result.duplicates_found} из {len(EXPECTED_DUPLICATES)} ожидаемых")

    print("\n📌 Для продакшена рекомендуется:")
    print("   - SemHash с threshold=0.85 (баланс скорости и точности)")
    print("   - Sentence-Transformers с мультиязычной моделью (для точности)")
    print("   - Гибридный подход: embeddings + LLM для edge cases")

    print("\n🔧 Настройка порога:")
    print("   - 0.95: консервативный (только явные дубликаты)")
    print("   - 0.85: сбалансированный ✅")
    print("   - 0.75: агрессивный (больше ложных срабатываний)")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
