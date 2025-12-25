"""Executor for LLMCompiler tasks."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool


class FunctionExecutor:
    """Executes functions with isolated memory spaces."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.memory = {}

    def execute(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        call_id: str,
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        try:
            self.memory[call_id] = {}
            result = tool.invoke(args, config)
            del self.memory[call_id]
            return result
        except Exception:
            self.memory.pop(call_id, None)
            raise


class ExecutorPool:
    """Pool of function executors for parallel task execution."""

    def __init__(self, num_executors: int = 4):
        self.executors = [FunctionExecutor() for _ in range(num_executors)]
        self.current = 0

    def get_executor(self) -> FunctionExecutor:
        executor = self.executors[self.current]
        self.current = (self.current + 1) % len(self.executors)
        return executor

    def execute_batch(
        self, tasks: List[Dict[str, Any]], config: Optional[RunnableConfig] = None
    ) -> List[Any]:
        results = []
        futures = []

        with ThreadPoolExecutor() as thread_pool:
            for task in tasks:
                executor = self.get_executor()
                future = thread_pool.submit(
                    executor.execute,
                    tool=task["tool"],
                    args=task["args"],
                    call_id=str(task["idx"]),
                    config=config,
                )
                futures.append(future)

        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(f"ERROR: {e}")

        return results
