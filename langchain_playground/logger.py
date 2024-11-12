import logging
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

log_queue = Queue()
queue_handler = QueueHandler(log_queue)

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(queue_handler)