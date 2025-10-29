"""Async utilities and helpers."""

import asyncio
from typing import Any, Awaitable, Callable, List, Optional, TypeVar

from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


async def gather_with_concurrency(
    n: int, *tasks: Awaitable[T], return_exceptions: bool = True
) -> List[T]:
    """
    Run tasks with a concurrency limit.

    Args:
        n: Maximum number of concurrent tasks
        *tasks: Async tasks to run
        return_exceptions: Whether to return exceptions or raise them

    Returns:
        List of results

    Example:
        >>> tasks = [process_doc(doc) for doc in documents]
        >>> results = await gather_with_concurrency(10, *tasks)
    """
    semaphore = asyncio.Semaphore(n)

    async def with_semaphore(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    wrapped_tasks = [with_semaphore(task) for task in tasks]
    return await asyncio.gather(*wrapped_tasks, return_exceptions=return_exceptions)


async def run_with_timeout(
    coro: Awaitable[T], timeout: float, default: Optional[T] = None
) -> T:
    """
    Run coroutine with timeout.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value to return on timeout

    Returns:
        Result or default value

    Raises:
        asyncio.TimeoutError: If timeout exceeded and no default provided
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if default is not None:
            logger.warning("operation_timeout", timeout=timeout)
            return default
        raise


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    **kwargs: Any,
) -> T:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        **kwargs: Keyword arguments for func

    Returns:
        Function result

    Raises:
        Exception: If all attempts fail

    Example:
        >>> result = await retry_async(
        ...     api_call,
        ...     param1,
        ...     max_attempts=3,
        ...     delay=1.0,
        ...     exceptions=(APIError, RateLimitError)
        ... )
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == max_attempts:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=max_attempts,
                    error=str(e),
                )
                raise

            wait_time = delay * (backoff ** (attempt - 1))
            logger.warning(
                "retry_attempt",
                function=func.__name__,
                attempt=attempt,
                max_attempts=max_attempts,
                wait_time=wait_time,
                error=str(e),
            )
            await asyncio.sleep(wait_time)

    # Should never reach here, but type checker needs it
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error")


class AsyncBatchProcessor:
    """Process items in batches asynchronously."""

    def __init__(
        self,
        batch_size: int = 100,
        max_concurrent: int = 10,
    ) -> None:
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_concurrent: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.logger = get_logger(__name__)

    async def process(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], Awaitable[List[T]]],
    ) -> List[T]:
        """
        Process items in batches.

        Args:
            items: Items to process
            process_func: Async function that processes a batch

        Returns:
            List of all results

        Example:
            >>> processor = AsyncBatchProcessor(batch_size=100, max_concurrent=5)
            >>> results = await processor.process(
            ...     all_texts,
            ...     lambda batch: embedding_provider.embed_texts(batch)
            ... )
        """
        if not items:
            return []

        # Create batches
        batches = [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]

        self.logger.info(
            "batch_processing_started",
            total_items=len(items),
            num_batches=len(batches),
            batch_size=self.batch_size,
        )

        # Process batches with concurrency limit
        batch_results = await gather_with_concurrency(
            self.max_concurrent, *[process_func(batch) for batch in batches]
        )

        # Flatten results
        results: List[T] = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error("batch_processing_error", error=str(batch_result))
                raise batch_result
            results.extend(batch_result)

        self.logger.info(
            "batch_processing_completed",
            total_items=len(items),
            total_results=len(results),
        )

        return results


class AsyncQueue:
    """Async queue with memory limits."""

    def __init__(
        self,
        maxsize: int = 0,
        max_memory_mb: Optional[float] = None,
    ) -> None:
        """
        Initialize async queue.

        Args:
            maxsize: Maximum number of items (0 = unlimited)
            max_memory_mb: Maximum memory usage in MB (None = unlimited)
        """
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0.0
        self._lock = asyncio.Lock()
        self.logger = get_logger(__name__)

    async def put(self, item: Any) -> None:
        """Put item in queue."""
        if self.max_memory_mb is not None:
            # Estimate item size (rough approximation)
            import sys

            item_size_mb = sys.getsizeof(item) / (1024 * 1024)

            async with self._lock:
                while self.current_memory_mb + item_size_mb > self.max_memory_mb:
                    self.logger.warning(
                        "queue_memory_limit",
                        current_mb=self.current_memory_mb,
                        max_mb=self.max_memory_mb,
                    )
                    await asyncio.sleep(0.1)  # Wait for space

                self.current_memory_mb += item_size_mb

        await self.queue.put(item)

    async def get(self) -> Any:
        """Get item from queue."""
        item = await self.queue.get()

        if self.max_memory_mb is not None:
            import sys

            item_size_mb = sys.getsizeof(item) / (1024 * 1024)
            async with self._lock:
                self.current_memory_mb -= item_size_mb

        return item

    def task_done(self) -> None:
        """Mark task as done."""
        self.queue.task_done()

    async def join(self) -> None:
        """Wait for all tasks to be done."""
        await self.queue.join()

    def qsize(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()


async def run_in_parallel(*coroutines: Awaitable[Any]) -> List[Any]:
    """
    Run multiple coroutines in parallel.

    Args:
        *coroutines: Coroutines to run

    Returns:
        List of results

    Example:
        >>> result1, result2 = await run_in_parallel(
        ...     chunk_document(doc1),
        ...     chunk_document(doc2)
        ... )
    """
    return await asyncio.gather(*coroutines)


def create_task_with_logging(
    coro: Awaitable[T], name: Optional[str] = None
) -> asyncio.Task[T]:
    """
    Create async task with automatic error logging.

    Args:
        coro: Coroutine to run as task
        name: Optional task name

    Returns:
        Created task
    """
    task = asyncio.create_task(coro, name=name)

    def log_exception(t: asyncio.Task[T]) -> None:
        try:
            t.result()
        except asyncio.CancelledError:
            logger.info("task_cancelled", task_name=name or "unnamed")
        except Exception as e:
            logger.error(
                "task_exception",
                task_name=name or "unnamed",
                error=str(e),
                exc_info=True,
            )

    task.add_done_callback(log_exception)
    return task
