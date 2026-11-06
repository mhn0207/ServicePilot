<template>
  <section class="rag-evaluation">
    <h2>RAG · Agent 离线验收</h2>
    <p>fixture/synthetic · scope=orchestrator · agent-dev-v1 · 固定 B / Top-K=3。模拟分数仅用于协议验收，真实质量与人工校准待验证。</p>
    <button :disabled="running" @click="execute(false)">开始离线验收</button>
    <button :disabled="running" @click="execute(true)">恢复已知任务</button>
    <button @click="history">刷新历史报告</button>
    <button v-if="state.includes('未知')" :disabled="running" @click="reconcile">已核对历史，解除未知提交</button>
    <p role="status">{{ state }}</p>
    <label>历史报告 <select v-model="selected" @change="load"><option value="">请选择</option><option v-for="run in runs" :key="run.run_id" :value="run.run_id">{{ run.run_id }} · {{ run.status }}</option></select></label>
    <div v-if="report">
      <p>run_id={{ report.run_id }} · {{ report.status }} · {{ report.scope }} · {{ report.mode }}</p>
      <details><summary>证据、标注和评分身份</summary><pre>{{ JSON.stringify(report.identities, null, 2) }}</pre></details>
      <table><thead><tr><th>检查</th><th>分子 / 分母</th><th>值 / 未决</th></tr></thead><tbody><tr v-for="(metric, name) in report.summary" :key="name"><td>{{ labels[name] || name }}</td><td>{{ metric.numerator ?? 'N/A' }} / {{ metric.denominator ?? 'N/A' }}</td><td>{{ metric.value == null ? 'N/A' : Number(metric.value).toFixed(3) }} · 未决 {{ metric.unresolved ?? 0 }}</td></tr></tbody></table>
      <p>检查为确定性观察；Judge 建议和未决不代表确定根因。Composer 只见角色文本时，直接证据 grounding 为 N/A。</p>
      <details v-for="turn in report.turns" :key="`${turn.case_id}:${turn.turn_id}`">
        <summary>{{ turn.case_id }} / {{ turn.turn_id }} · {{ turn.question }}</summary>
        <p>角色：{{ turn.roles }} → 实际查询：{{ turn.queries }}</p>
        <p>回答：{{ turn.answer }}</p>
        <details><summary>资料与工具事件</summary><pre>{{ JSON.stringify(turn.events, null, 2) }}</pre></details>
        <details><summary>实际生成输入与角色输出</summary><pre>{{ JSON.stringify({inputs:turn.generation_inputs,roles:turn.role_answers,steps:turn.model_steps}, null, 2) }}</pre></details>
        <details><summary>逐事实判定与实际证据支持</summary><pre>{{ JSON.stringify({correctness:turn.correctness,grounding:turn.grounding,role_grounding:turn.role_grounding}, null, 2) }}</pre></details>
        <pre>{{ JSON.stringify(turn.checks, null, 2) }}</pre>
      </details>
    </div>
  </section>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { createRequestState } from '../lib/requestState.js'
import { requestRagRuns, runRagEvaluation, clearUnknownRagSubmission } from '../lib/api.js'
const props = defineProps({settings: {type: Object, required: true}})
const requests = createRequestState(props.settings)
const labels = {required_call_coverage:'必需检索调用覆盖', execution_success:'检索执行成功', query_entities:'实际查询必要实体', old_entity_clean:'旧订单污染检查', returned_evidence:'返回证据覆盖', role_coverage:'必要角色覆盖', required_facts:'必要事实覆盖（轮次）', forbidden_clear:'无禁止结论（轮次）', judge_validity:'Judge 有效性（固定模拟）'}
const report = ref(null), runs = ref([]), selected = ref(''), state = ref('仅明确启动时提交新任务'), running = ref(false)
watch(() => [props.settings.endpoint, props.settings.userId, props.settings.pythonToken], () => {
  requests.invalidate(); report.value = null; runs.value = []; selected.value = ''; running.value = false; state.value = '连接已切换'; history()
})
onMounted(() => { history(); execute(true) })
onBeforeUnmount(() => requests.invalidate())
function errorText(error) { return error.status === 403 ? '权限不足：RAG 评测需要管理员权限' : error.status === 401 ? '访问令牌无效，请重新配置连接' : error.message }
async function reconcile() {
  const request = requests.begin('task')
  await clearUnknownRagSubmission(request.settings)
  if (request.isCurrent()) state.value = '已解除未知绑定。确认原任务状态后，可明确开始新任务。'
}
async function execute(resume) {
  if (running.value) return
  requests.begin('detail')
  const request = requests.begin('task'); running.value = true
  try {
    const result = await runRagEvaluation(request.settings, {isCurrent: request.isCurrent, onProgress: job => { if (request.isCurrent()) state.value = `${job.status} · ${job.completed}/${job.total}` }}, resume)
    if (!request.isCurrent()) return
    if (result) { report.value = result; state.value = `完成：${result.status}`; history() }
  } catch (error) { if (request.isCurrent()) state.value = errorText(error) }
  finally { if (request.isCurrent()) running.value = false }
}
async function history() {
  const request = requests.begin('history')
  try { const data = await requestRagRuns(request.settings); if (request.isCurrent()) runs.value = data }
  catch (error) { if (request.isCurrent()) state.value = errorText(error) }
}
async function load() {
  if (!selected.value) return
  requests.begin('task'); running.value = false
  const request = requests.begin('detail')
  try { const data = await requestRagRuns(request.settings, selected.value); if (request.isCurrent()) report.value = data }
  catch (error) { if (request.isCurrent()) state.value = errorText(error) }
}
</script>

<style scoped>
.rag-evaluation { margin-bottom: 28px; padding: 24px; border: 1px solid #dce2eb; border-radius: 14px; background: #fff; }
button, select { margin: 8px; padding: 8px; }
table { width: 100%; text-align: left; margin: 18px 0; }
th, td { padding: 8px; border-bottom: 1px solid #eee; }
details { margin: 12px 0; } summary { cursor: pointer; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 420px; overflow: auto; font-size: 12px; }
</style>
