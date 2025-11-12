"""
Telegram Message Deduplicator
Семантическая дедупликация сообщений из нескольких Telegram каналов

Требования:
    pip install semhash telethon python-dotenv

Использование:
    1. Создайте .env файл с API_ID, API_HASH, PHONE
    2. Запустите скрипт: python telegram_deduplicator.py
"""

import asyncio
import os
from typing import List, Set
from datetime import datetime
from telethon import TelegramClient, events
from semhash import SemHash
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')

# ID каналов (замените на ваши)
SOURCE_CHANNELS = [
    # -1001234567890,  # Канал 1
    # -1009876543210,  # Канал 2
]

TARGET_CHANNEL = None  # -1001111111111  # Целевой канал

# Порог семантической схожести (0.0 - 1.0)
SIMILARITY_THRESHOLD = 0.85

# Размер окна для дедупликации (последние N сообщений)
DEDUP_WINDOW_SIZE = 1000


class TelegramDeduplicator:
    """Класс для дедупликации сообщений из множества Telegram каналов"""

    def __init__(
        self,
        api_id: str,
        api_hash: str,
        phone: str,
        source_channels: List[int],
        target_channel: int,
        similarity_threshold: float = 0.85,
        window_size: int = 1000
    ):
        self.client = TelegramClient('deduplicator_session', api_id, api_hash)
        self.source_channels = source_channels
        self.target_channel = target_channel
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

        # Хранилище сообщений
        self.message_history: List[str] = []
        self.processed_ids: Set[int] = set()

        # Статистика
        self.stats = {
            'received': 0,
            'forwarded': 0,
            'duplicates': 0
        }

    async def start(self):
        """Запуск бота"""
        await self.client.start(phone=PHONE)
        logger.info("🚀 Deduplicator запущен!")

        # Загрузка истории из целевого канала
        await self.load_history()

        # Регистрация обработчика новых сообщений
        @self.client.on(events.NewMessage(chats=self.source_channels))
        async def handle_new_message(event):
            await self.process_message(event)

        logger.info(f"👂 Слушаем {len(self.source_channels)} каналов...")
        logger.info(f"🎯 Целевой канал: {self.target_channel}")
        logger.info(f"📊 Порог схожести: {self.similarity_threshold}")

        # Запуск клиента
        await self.client.run_until_disconnected()

    async def load_history(self):
        """Загрузка истории сообщений из целевого канала"""
        if not self.target_channel:
            logger.warning("⚠️ Целевой канал не указан, пропускаем загрузку истории")
            return

        logger.info("📥 Загрузка истории сообщений из целевого канала...")

        try:
            messages = await self.client.get_messages(
                self.target_channel,
                limit=self.window_size
            )

            for msg in messages:
                if msg.text:
                    self.message_history.append(msg.text)

            logger.info(f"✅ Загружено {len(self.message_history)} сообщений из истории")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки истории: {e}")

    async def process_message(self, event):
        """Обработка нового сообщения"""
        message = event.message

        # Пропускаем сообщения без текста
        if not message.text:
            return

        # Пропускаем уже обработанные
        if message.id in self.processed_ids:
            return

        self.processed_ids.add(message.id)
        self.stats['received'] += 1

        message_text = message.text

        # Получение информации о канале
        chat = await event.get_chat()
        channel_name = getattr(chat, 'title', 'Unknown')

        logger.info(f"📨 Новое сообщение из '{channel_name}': {message_text[:50]}...")

        # Проверка на дубликат
        is_unique = self.check_uniqueness(message_text)

        if is_unique:
            # Уникальное сообщение - пересылаем
            await self.forward_message(message_text, channel_name)
            self.stats['forwarded'] += 1

            # Добавляем в историю
            self.message_history.append(message_text)

            # Ограничиваем размер истории
            if len(self.message_history) > self.window_size:
                self.message_history = self.message_history[-self.window_size:]

        else:
            # Дубликат - игнорируем
            self.stats['duplicates'] += 1
            logger.info(f"❌ Дубликат обнаружен (схожесть > {self.similarity_threshold})")

        # Периодический вывод статистики
        if self.stats['received'] % 10 == 0:
            self.print_stats()

    def check_uniqueness(self, text: str) -> bool:
        """
        Проверка уникальности сообщения

        Returns:
            True если сообщение уникально, False если дубликат
        """
        if not self.message_history:
            return True

        try:
            # Создание индекса из истории
            deduplicator = SemHash.from_records(
                records=self.message_history,
                threshold=self.similarity_threshold
            )

            # Проверка нового сообщения
            result = deduplicator.deduplicate(records=[text])

            # Если сообщение в результате - оно уникально
            return len(result.selected) > 0

        except Exception as e:
            logger.error(f"❌ Ошибка при проверке уникальности: {e}")
            # В случае ошибки считаем сообщение уникальным
            return True

    async def forward_message(self, text: str, source: str):
        """Пересылка уникального сообщения в целевой канал"""
        if not self.target_channel:
            logger.info(f"✅ [TEST MODE] Уникальное сообщение из '{source}': {text[:100]}...")
            return

        try:
            # Добавляем источник к сообщению (опционально)
            formatted_text = f"📢 {text}\n\n└ Источник: {source}"

            await self.client.send_message(
                self.target_channel,
                formatted_text
            )

            logger.info(f"✅ Сообщение переслано в целевой канал")

        except Exception as e:
            logger.error(f"❌ Ошибка при пересылке: {e}")

    def print_stats(self):
        """Вывод статистики"""
        total = self.stats['received']
        forwarded = self.stats['forwarded']
        duplicates = self.stats['duplicates']

        dup_rate = (duplicates / total * 100) if total > 0 else 0

        logger.info("=" * 60)
        logger.info("📊 СТАТИСТИКА")
        logger.info(f"   Получено сообщений: {total}")
        logger.info(f"   ✅ Уникальных: {forwarded} ({100-dup_rate:.1f}%)")
        logger.info(f"   ❌ Дубликатов: {duplicates} ({dup_rate:.1f}%)")
        logger.info(f"   📝 В истории: {len(self.message_history)}")
        logger.info("=" * 60)


async def main():
    """Главная функция"""

    # Проверка конфигурации
    if not API_ID or not API_HASH or not PHONE:
        logger.error("❌ Заполните .env файл с API_ID, API_HASH и PHONE")
        return

    if not SOURCE_CHANNELS:
        logger.warning("⚠️ SOURCE_CHANNELS пуст. Добавьте ID исходных каналов.")
        logger.info("💡 Для получения ID канала используйте: https://t.me/username_to_id_bot")
        return

    # Создание и запуск дедупликатора
    deduplicator = TelegramDeduplicator(
        api_id=API_ID,
        api_hash=API_HASH,
        phone=PHONE,
        source_channels=SOURCE_CHANNELS,
        target_channel=TARGET_CHANNEL,
        similarity_threshold=SIMILARITY_THRESHOLD,
        window_size=DEDUP_WINDOW_SIZE
    )

    try:
        await deduplicator.start()
    except KeyboardInterrupt:
        logger.info("\n👋 Остановка бота...")
        deduplicator.print_stats()
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == '__main__':
    asyncio.run(main())
