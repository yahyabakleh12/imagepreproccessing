# plate_runtime.py
import uuid
import time
import queue
from multiprocessing import Process, Queue
from typing import Optional

from plate_worker import plate_worker
from logger import get_logger

logger = get_logger("plate_runtime", "plate_runtime.log")

# Global references for worker process and queues
_plate_proc: Optional[Process] = None
_job_queue: Optional[Queue] = None
_result_queue: Optional[Queue] = None


def start_plate_worker():
    """Start the plate detection worker process if not already started."""
    global _plate_proc, _job_queue, _result_queue

    # If already running, do nothing
    if _plate_proc is not None and _plate_proc.is_alive():
        return

    logger.info("Starting plate worker process...")

    _job_queue = Queue()
    _result_queue = Queue()
    _plate_proc = Process(
        target=plate_worker,
        args=(_job_queue, _result_queue),
        daemon=True,
    )
    _plate_proc.start()

    logger.info("Plate worker started with PID=%s", _plate_proc.pid)


def stop_plate_worker():
    """Send STOP signal to plate worker and wait for it to exit."""
    global _plate_proc, _job_queue, _result_queue

    if _plate_proc is None:
        return

    try:
        if _job_queue is not None:
            _job_queue.put("STOP")
        _plate_proc.join(timeout=5)
        logger.info("Plate worker process stopped.")
    except Exception as e:
        logger.error("Error while stopping plate worker: %s", e)

    _plate_proc = None
    _job_queue = None
    _result_queue = None


def has_plate(image_path: str, timeout: float = 3.0) -> bool:
    """
    Synchronously check if an image has a plate, using the plate worker process.

    - Starts the worker lazily on first call.
    - Sends a job (image_path, job_id) to the worker.
    - Waits up to `timeout` seconds for the result.
    - Returns True if worker says has_plate=True, otherwise False.
    """
    global _plate_proc, _job_queue, _result_queue

    # Ensure worker is running
    if _plate_proc is None or not _plate_proc.is_alive():
        start_plate_worker()

    if _job_queue is None or _result_queue is None:
        logger.error("Plate worker queues are not initialized.")
        return False

    job_id = str(uuid.uuid4())
    try:
        _job_queue.put((image_path, job_id))
    except Exception as e:
        logger.error("Failed to send job to plate worker: %s", e)
        return False

    start_t = time.time()
    while True:
        remaining = timeout - (time.time() - start_t)
        if remaining <= 0:
            logger.warning("Timeout waiting for plate worker result job_id=%s", job_id)
            return False

        try:
            res_job_id, has_plate_flag = _result_queue.get(timeout=remaining)
            if res_job_id == job_id:
                logger.info(
                    "Plate worker result job_id=%s has_plate=%s",
                    job_id,
                    has_plate_flag,
                )
                return bool(has_plate_flag)
            else:
                # Very rare: some other job result comes first
                logger.warning(
                    "Received mismatched job_id from plate worker: expected=%s got=%s",
                    job_id,
                    res_job_id,
                )
        except queue.Empty:
            logger.warning(
                "Result queue empty (timeout) for plate worker job_id=%s", job_id
            )
            return False
        except Exception as e:
            logger.error("Error while reading plate worker result: %s", e)
            return False
