// 对连接配置和请求代次做快照，避免旧请求覆盖当前页面。
export function createRequestState(settings) {
  let generation = 0
  let sequence = 0
  const latest = new Map()
  return {
    invalidate() { generation += 1 },
    begin(slot) {
      const revision = generation
      const id = ++sequence
      const snapshot = { ...settings }
      latest.set(slot, id)
      return {
        settings: snapshot,
        isCurrent: () => revision === generation && latest.get(slot) === id &&
          snapshot.endpoint === settings.endpoint && snapshot.userId === settings.userId &&
          snapshot.pythonToken === settings.pythonToken
      }
    }
  }
}
