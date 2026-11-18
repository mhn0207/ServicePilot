"""Share the knowledge cache generation across API workers."""

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from redis.exceptions import RedisError

from core.concurrency import redis_lease


class KnowledgeRevision:
    key = "servicepilot:knowledge:revision:v1"

    def __init__(self, redis):
        self.redis = redis

    async def current(self) -> str | None:
        try:
            async with asyncio.timeout(.5):
                value = await self.redis.get(self.key)
                if value is None:
                    await self.redis.set(self.key, uuid4().hex, nx=True)
                    value = await self.redis.get(self.key)
        except (RedisError, TimeoutError):
            return None
        return value if value and not value.startswith("dirty:") else None

    @asynccontextmanager
    async def mutation(self):
        async with redis_lease(self.redis, f"lock:{self.key}", wait_s=5):
            revision = uuid4().hex
            marker = f"dirty:{revision}"
            await self.redis.set(self.key, marker)
            try:
                yield
            except Exception:
                # A multi-step Chroma write may have changed data before failing.
                await self._publish(marker, revision)
                raise
            else:
                await self._publish(marker, revision)

    async def _publish(self, marker, revision):
        await self.redis.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('set', KEYS[1], ARGV[2]) end",
            1, self.key, marker, revision,
        )
