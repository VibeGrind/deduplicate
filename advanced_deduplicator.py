"""
Advanced Telegram Message Deduplicator
Расширенная версия с поддержкой:
- Sentence Transformers для кастомных моделей
- Qdrant для персистентности
- LLM-верификация для edge cases
- Мультиязычная поддержка

Требования:
    pip install sentence-transformers qdrant-client openai telethon python-dotenv
"""

import asyncio
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from sentence_transformers import SentenceTransformer, util
from telethon import TelegramClient, events
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import logging
import torch

# Опционально: для LLM-верификации
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Конфигурация
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

SOURCE_CHANNELS = []  # Заполните
TARGET_CHANNEL = None

# Настройки дедупликации
SIMILARITY_THRESHOLD = 0.85
EDGE_CASE_THRESHOLD = 0.75  # Для LLM-верификации
USE_LLM_VERIFICATION = False  # Включить для edge cases


@dataclass
class Message:
    """Структура сообщения"""
    id: int
    text: str
    channel: str
    timestamp: datetime
    embedding: Optional[List[float]] = None


class AdvancedDeduplicator:
    """Продвинутый дедупликатор с поддержкой векторной БД и LLM"""

    def __init__(
        self,
        api_id: str,
        api_hash: str,
        phone: str,
        source_channels: List[int],
        target_channel: Optional[int],
        model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
        similarity_threshold: float = 0.85,
        use_qdrant: bool = True,
        use_llm_verification: bool = False
    ):
        self.client = TelegramClient('advanced_session', api_id, api_hash)
        self.source_channels = source_channels
        self.target_channel = target_channel
        self.similarity_threshold = similarity_threshold
        self.use_llm_verification = use_llm_verification

        # Загрузка модели для embeddings
        logger.info(f"📥 Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("✅ Модель загружена")

        # Инициализация Qdrant (опционально)
        self.use_qdrant = use_qdrant
        if use_qdrant:
            self.qdrant_client = QdrantClient(":memory:")  # Или указать путь к БД
            self._init_qdrant()
        else:
            # Хранилище в памяти
            self.message_embeddings = []
            self.message_texts = []

        # OpenAI клиент для LLM-верификации
        if use_llm_verification and OPENAI_AVAILABLE and OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.openai_client = None

        # Статистика
        self.stats = {
            'received': 0,
            'forwarded': 0,
            'duplicates_embedding': 0,
            'duplicates_llm': 0,
            'llm_verifications': 0
        }

        self.message_counter = 0

    def _init_qdrant(self):
        """Инициализация Qdrant коллекции"""
        collection_name = "telegram_messages"

        # Получаем размерность вектора из модели
        vector_size = self.model.get_sentence_embedding_dimension()

        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Qdrant коллекция '{collection_name}' создана (размерность: {vector_size})")
        except Exception as e:
            logger.info(f"⚠️ Коллекция уже существует или ошибка: {e}")

        self.collection_name = collection_name

    async def start(self):
        """Запуск бота"""
        await self.client.start(phone=PHONE)
        logger.info("🚀 Advanced Deduplicator запущен!")
        logger.info(f"🧠 Модель: {self.model._model_card_data.get('model_name', 'Unknown')}")
        logger.info(f"🗄️ Хранилище: {'Qdrant' if self.use_qdrant else 'In-Memory'}")
        logger.info(f"🤖 LLM-верификация: {'Включена' if self.openai_client else 'Отключена'}")

        # Загрузка истории
        await self.load_history()

        # Регистрация обработчика
        @self.client.on(events.NewMessage(chats=self.source_channels))
        async def handle_new_message(event):
            await self.process_message(event)

        logger.info(f"👂 Слушаем {len(self.source_channels)} каналов...")

        await self.client.run_until_disconnected()

    async def load_history(self):
        """Загрузка истории сообщений"""
        if not self.target_channel:
            logger.warning("⚠️ Целевой канал не указан")
            return

        logger.info("📥 Загрузка истории...")

        try:
            messages = await self.client.get_messages(self.target_channel, limit=500)

            for msg in messages:
                if msg.text:
                    embedding = self.model.encode(msg.text, convert_to_tensor=False)
                    self._add_to_index(msg.text, embedding)

            logger.info(f"✅ Загружено {len(messages)} сообщений")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки истории: {e}")

    async def process_message(self, event):
        """Обработка нового сообщения"""
        message = event.message

        if not message.text:
            return

        self.stats['received'] += 1
        message_text = message.text

        # Получение информации о канале
        chat = await event.get_chat()
        channel_name = getattr(chat, 'title', 'Unknown')

        logger.info(f"📨 [{channel_name}] {message_text[:60]}...")

        # Проверка уникальности
        is_unique, similarity, similar_text = await self.check_uniqueness(message_text)

        if is_unique:
            # Уникальное сообщение
            await self.forward_message(message_text, channel_name)
            self.stats['forwarded'] += 1

            # Добавление в индекс
            embedding = self.model.encode(message_text, convert_to_tensor=False)
            self._add_to_index(message_text, embedding)

        else:
            # Дубликат
            if similarity >= self.similarity_threshold:
                self.stats['duplicates_embedding'] += 1
                logger.info(f"❌ Дубликат (embedding similarity: {similarity:.3f})")
            else:
                self.stats['duplicates_llm'] += 1
                logger.info(f"❌ Дубликат (LLM verification)")

            if similar_text:
                logger.info(f"   └ Похож на: {similar_text[:60]}...")

        # Периодическая статистика
        if self.stats['received'] % 10 == 0:
            self.print_stats()

    async def check_uniqueness(self, text: str) -> Tuple[bool, float, Optional[str]]:
        """
        Проверка уникальности сообщения

        Returns:
            (is_unique, similarity_score, similar_text)
        """
        # Генерация embedding
        embedding = self.model.encode(text, convert_to_tensor=True)

        # Поиск похожих сообщений
        if self.use_qdrant:
            similar_text, max_similarity = self._search_qdrant(embedding.cpu().numpy().tolist())
        else:
            similar_text, max_similarity = self._search_memory(embedding)

        logger.debug(f"   Max similarity: {max_similarity:.3f}")

        # Случай 1: Явно уникальное
        if max_similarity < EDGE_CASE_THRESHOLD:
            return True, max_similarity, None

        # Случай 2: Явно дубликат
        if max_similarity >= self.similarity_threshold:
            return False, max_similarity, similar_text

        # Случай 3: Пограничный случай - проверка через LLM
        if self.openai_client and self.use_llm_verification:
            logger.info(f"🤔 Пограничный случай (similarity: {max_similarity:.3f}), проверяем через LLM...")
            self.stats['llm_verifications'] += 1

            is_duplicate = await self._llm_verify_duplicate(text, similar_text)

            if is_duplicate:
                return False, max_similarity, similar_text

        # По умолчанию считаем уникальным
        return True, max_similarity, similar_text

    def _search_qdrant(self, embedding: List[float]) -> Tuple[Optional[str], float]:
        """Поиск в Qdrant"""
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=1
        )

        if results:
            return results[0].payload['text'], results[0].score

        return None, 0.0

    def _search_memory(self, embedding: torch.Tensor) -> Tuple[Optional[str], float]:
        """Поиск в памяти"""
        if not self.message_embeddings:
            return None, 0.0

        # Вычисление косинусного сходства
        embeddings_tensor = torch.stack(self.message_embeddings)
        similarities = util.cos_sim(embedding, embeddings_tensor)[0]

        max_similarity = similarities.max().item()
        max_idx = similarities.argmax().item()

        return self.message_texts[max_idx], max_similarity

    def _add_to_index(self, text: str, embedding):
        """Добавление сообщения в индекс"""
        if self.use_qdrant:
            # Добавление в Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=self.message_counter,
                        vector=embedding.tolist() if isinstance(embedding, torch.Tensor) else embedding,
                        payload={"text": text}
                    )
                ]
            )
        else:
            # Добавление в память
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding)

            self.message_embeddings.append(embedding)
            self.message_texts.append(text)

        self.message_counter += 1

    async def _llm_verify_duplicate(self, text1: str, text2: str) -> bool:
        """LLM-верификация дубликата для пограничных случаев"""
        prompt = f"""Are these two messages semantically identical (conveying the same information)?

Message 1: {text1}

Message 2: {text2}

Consider:
- Different wording but same meaning = YES
- Similar topic but different details = NO

Answer only: YES or NO"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Дешевая модель
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )

            answer = response.choices[0].message.content.strip().upper()
            logger.info(f"   🤖 LLM ответ: {answer}")

            return answer == "YES"

        except Exception as e:
            logger.error(f"❌ Ошибка LLM-верификации: {e}")
            return False

    async def forward_message(self, text: str, source: str):
        """Пересылка уникального сообщения"""
        if not self.target_channel:
            logger.info(f"✅ [TEST] Уникальное: {text[:80]}...")
            return

        try:
            await self.client.send_message(self.target_channel, text)
            logger.info(f"✅ Переслано")

        except Exception as e:
            logger.error(f"❌ Ошибка пересылки: {e}")

    def print_stats(self):
        """Статистика"""
        total = self.stats['received']
        forwarded = self.stats['forwarded']
        dup_emb = self.stats['duplicates_embedding']
        dup_llm = self.stats['duplicates_llm']
        llm_checks = self.stats['llm_verifications']

        total_duplicates = dup_emb + dup_llm
        dup_rate = (total_duplicates / total * 100) if total > 0 else 0

        logger.info("=" * 70)
        logger.info("📊 СТАТИСТИКА")
        logger.info(f"   Получено: {total}")
        logger.info(f"   ✅ Уникальных: {forwarded} ({100-dup_rate:.1f}%)")
        logger.info(f"   ❌ Дубликатов (embedding): {dup_emb}")
        logger.info(f"   ❌ Дубликатов (LLM): {dup_llm}")
        logger.info(f"   🤖 LLM-проверок: {llm_checks}")
        logger.info(f"   📝 В индексе: {self.message_counter}")
        logger.info("=" * 70)


async def main():
    """Главная функция"""

    if not API_ID or not API_HASH or not PHONE:
        logger.error("❌ Заполните .env файл")
        return

    if not SOURCE_CHANNELS:
        logger.warning("⚠️ Добавьте SOURCE_CHANNELS")
        return

    # Выбор модели
    # Мультиязычные модели:
    # - 'paraphrase-multilingual-mpnet-base-v2' (рекомендуется)
    # - 'distiluse-base-multilingual-cased-v2' (быстрее)
    # Английские:
    # - 'all-MiniLM-L6-v2' (быстрая)
    # - 'all-mpnet-base-v2' (точная)

    deduplicator = AdvancedDeduplicator(
        api_id=API_ID,
        api_hash=API_HASH,
        phone=PHONE,
        source_channels=SOURCE_CHANNELS,
        target_channel=TARGET_CHANNEL,
        model_name='paraphrase-multilingual-mpnet-base-v2',  # Для русского + английского
        similarity_threshold=SIMILARITY_THRESHOLD,
        use_qdrant=True,  # Персистентность
        use_llm_verification=USE_LLM_VERIFICATION  # LLM для edge cases
    )

    try:
        await deduplicator.start()
    except KeyboardInterrupt:
        logger.info("\n👋 Остановка...")
        deduplicator.print_stats()
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == '__main__':
    asyncio.run(main())
