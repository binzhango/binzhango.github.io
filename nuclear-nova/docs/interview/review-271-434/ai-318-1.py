from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class ToolSpec:
    name: str
    input_schema: dict
    output_schema: dict
    side_effecting: bool

@dataclass(frozen=True)
class CallContext:
    principal: str
    deadline: float
    idempotency_key: str

class ToolAdapter(Protocol):
    spec: ToolSpec
    async def invoke(self, arguments: dict, context: CallContext) -> Any: ...
