"""回答生成和两个 Judge 阶段的独立依赖，不导入线上状态或 dotenv。"""
import os
from pathlib import Path
from urllib.parse import urlsplit

from evaluation.experiments import load_json
from evaluation.retrieval import canonical_hash

DEFAULT_FIXTURE = Path(__file__).parent / 'fixtures/answers.v4.json'


class FixedModel:
    mode = 'fixture'

    def __init__(self, path=DEFAULT_FIXTURE):
        self.fixture = load_json(path)
        if self.fixture.get('protocol_version') != 'rag-answer-fixture-v1':
            raise ValueError('invalid answer fixture')
        payload = {k:v for k,v in self.fixture.items() if k != 'fixture_hash'}
        if canonical_hash(payload) != self.fixture['fixture_hash']:
            raise ValueError('fixture hash mismatch')
        self.identity = 'fixed-answer-and-judge-v1:' + self.fixture['fixture_hash']

    def validate_inputs(self, suite):
        from evaluation.answer_quality import protocol_identity
        if (self.fixture.get('annotation_hash') != suite[0].annotation_hash
                or self.fixture.get('knowledge_hash') != suite[2].knowledge_hash
                or self.fixture.get('prompts') not in (protocol_identity(), protocol_identity(span_coverage=False), protocol_identity(legacy=True))):
            raise ValueError('fixture input identity mismatch')

    async def call(self, stage, payload, parameters):
        # 精确请求哈希匹配：不是语义Judge，也不依据运行中的参考临时给分。
        return self.fixture['responses'][stage][canonical_hash(payload)]


class AnswerProvider:
    mode = 'real'

    def __init__(self, role, client=None, model=None, endpoint=None):
        from anthropic import AsyncAnthropic
        prefix = 'SERVICEPILOT_ANSWER_' + role.upper()
        self.model = model or os.environ.get(prefix + '_MODEL')
        key = os.environ.get(prefix + '_API_KEY')
        url = endpoint or os.environ.get(prefix + '_BASE_URL', 'https://api.anthropic.com')
        parsed = urlsplit(url)
        if (parsed.scheme not in ('https', 'http') or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('unsafe provider endpoint')
        if not self.model or (client is None and not key):
            raise ValueError('explicit provider process environment required')
        self.identity = self.model + ':endpoint-' + canonical_hash(url)
        self.client = client or AsyncAnthropic(api_key=key, base_url=url, max_retries=0, timeout=45)
        self.deepseek = parsed.hostname == 'api.deepseek.com'

    async def call(self, stage, payload, parameters):
        extra = {'extra_body': {'thinking': {'type': 'disabled'}}} if self.deepseek else {}
        response = await self.client.messages.create(model=self.model, **parameters, **payload, **extra)
        raw = ''.join(b.text for b in response.content if getattr(b, 'type', None) == 'text')
        usage = getattr(response, 'usage', None)
        return dict(text=raw, stop_reason=response.stop_reason, **{field:getattr(usage, field, None) for field in (
            'input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')})

    async def close(self):
        await self.client.close()
