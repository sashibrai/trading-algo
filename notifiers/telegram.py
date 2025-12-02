import os
import requests
import logging

def send_telegram_message(message: str):
    """
    Sends a message to the configured Telegram chats.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_ids_str = os.getenv("TELEGRAM_CHAT_IDS")

    if not bot_token or not chat_ids_str:
        logging.warning("Telegram bot token or chat IDs not configured. Skipping notification.")
        return

    chat_ids = [chat_id.strip() for chat_id in chat_ids_str.split(",")]

    for chat_id in chat_ids:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                logging.error(f"Error sending Telegram message to chat ID {chat_id}: {response.text}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while sending Telegram message: {e}")
