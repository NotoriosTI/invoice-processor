import logging
import threading
from typing import List, Dict, Any

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

logger = logging.getLogger(__name__)

ALLOWED_INVOICE_FILETYPES = {"png", "jpg", "jpeg", "bmp", "gif", "pdf"}


class SlackBot:
    def __init__(
        self,
        message_queue,
        bot_token: str,
        app_token: str,
        debug: bool = False,
    ):
        self.message_queue = message_queue
        self.debug = debug
        self.app = App(token=bot_token)
        self.socket_token = app_token
        self._register_events()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _add_reaction(self, channel: str | None, timestamp: str | None, emoji: str = "inbox_tray") -> None:
        if not channel or not timestamp:
            return
        try:
            self.app.client.reactions_add(channel=channel, timestamp=timestamp, name=emoji)
        except Exception as exc:
            logger.debug("No se pudo agregar reacción en Slack: %s", exc)

    @staticmethod
    def _extract_invoice_files(files: List[dict]) -> List[dict]:
        invoice_files: List[dict] = []
        for file in files or []:
            filetype = (file.get("filetype") or "").lower()
            if filetype in ALLOWED_INVOICE_FILETYPES:
                invoice_files.append(
                    {
                        "id": file.get("id"),
                        "name": file.get("name"),
                        "url_private_download": file.get("url_private_download"),
                        "filetype": filetype,
                    }
                )
        return invoice_files

    def _enqueue_invoice_event(
        self,
        user_id: str,
        files: List[dict],
        text: str | None,
        channel: str | None,
        timestamp: str | None,
    ) -> None:
        if not files:
            return
        payload = {
            "user": user_id,
            "text": text or "",
            "files": files,
            "channel": channel,
            "timestamp": timestamp,
        }
        if self.debug:
            logger.info("Encolando evento de Slack: %s", payload)
        self.message_queue.put(payload)

    # ------------------------------------------------------------------ #
    # Event handlers                                                      #
    # ------------------------------------------------------------------ #

    def _register_events(self) -> None:
        @self.app.event("message")
        def handle_message_events(event, say):
            user_id = event.get("user")
            channel = event.get("channel")
            timestamp = event.get("ts")

            invoice_files = self._extract_invoice_files(event.get("files", []))
            if invoice_files:
                self._add_reaction(channel, timestamp)
                self._enqueue_invoice_event(user_id, invoice_files, event.get("text", ""), channel, timestamp)
                return

            text = (event.get("text") or "").strip()
            if text:
                self._add_reaction(channel, timestamp, emoji="speech_balloon")
                payload = {
                    "user": user_id,
                    "text": text,
                    "say": say,
                    "files": [],
                    "channel": channel,
                    "timestamp": timestamp,
                }
                self.message_queue.put(payload)

        @self.app.event("file_shared")
        def handle_file_shared(event, say):
            file_id = event.get("file_id")
            user_id = event.get("user_id")
            channel = event.get("channel_id")
            timestamp = event.get("message_ts")
            if not file_id or not user_id:
                return
            try:
                file_info = self.app.client.files_info(file=file_id)["file"]
            except Exception as exc:
                logger.error("No se pudo obtener info del archivo %s: %s", file_id, exc)
                return

            invoice_files = self._extract_invoice_files([file_info])
            if invoice_files:
                self._add_reaction(channel, timestamp)
                self._enqueue_invoice_event(user_id, invoice_files, "", channel, timestamp)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        handler = SocketModeHandler(self.app, self.socket_token)
        thread = threading.Thread(target=handler.start, daemon=True)
        thread.start()
