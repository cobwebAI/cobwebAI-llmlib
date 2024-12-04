import logging
from logging.handlers import QueueHandler
from queue import Queue

log_queue = Queue()
queue_handler = QueueHandler(log_queue)

log = logging.getLogger("langchain_playground")
log.setLevel(logging.DEBUG)
log.addHandler(queue_handler)