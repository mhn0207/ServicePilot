"""Bounded, read-only storage probes isolated from Chroma's sync HTTP pool."""
import asyncio
from urllib.parse import quote, urlsplit

import httpx


class ChromaHealthProbe:
    def __init__(self, collection, *, host=None, port=8000, api_path='/api/v2'):
        self._count = collection.count
        self._pending = None
        self._url = None
        if host is not None:
            if host.startswith(('http://', 'https://')):
                base = host.rstrip('/')
            else:
                address = f'[{host}]' if ':' in host else host
                base = f'http://{address}:{port}'
            # Use the running client's API version, including proxy prefixes.
            if not urlsplit(base).path.endswith(api_path):
                base += api_path
            tenant, database, identity = (quote(str(value), safe='') for value in
                                          (collection.tenant, collection.database, collection.id))
            self._url = (f'{base}/collections/{identity}/count' if api_path == '/api/v1' else
                         f'{base}/tenants/{tenant}/databases/{database}/collections/{identity}/count')

    async def check(self) -> bool:
        # Cancellation of /health must not spawn another local disk thread on
        # the next probe. Remote probes are async and have real I/O timeouts.
        if self._pending is None or self._pending.done():
            self._pending = asyncio.create_task(self._probe())
        return await asyncio.shield(self._pending)

    async def _probe(self) -> bool:
        try:
            if self._url is not None:
                async with httpx.AsyncClient(timeout=1.0, trust_env=False) as client:
                    response = await client.get(self._url)
                    response.raise_for_status()
                    payload = response.json()
                    return type(payload) is int and payload >= 0
            count = await asyncio.to_thread(self._count)
            return type(count) is int and count >= 0
        except Exception:
            return False
