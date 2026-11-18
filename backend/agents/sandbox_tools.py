"""账单 Agent 的只读工具，以及用户直接授权的沙盒聊天命令。"""

import asyncio
import re
import sqlite3
from dataclasses import replace

from agents.tools import make_tool
from tooling.sandbox_store import DEMO_ORDER_ID, SandboxStore


ORDER_PATTERN = r"ORD-[0-9]{8}"
REVIEW_PATTERN = r"RFD-[0-9A-F]{32}"
ORDER_LOOKUP = "query_sandbox_order_ledger"
COMMANDS = {
    "沙盒查账": (ORDER_PATTERN, ORDER_LOOKUP, DEMO_ORDER_ID),
    "确认提交退款核验": (ORDER_PATTERN, "submit_sandbox_refund_review", DEMO_ORDER_ID),
    "查询退款核验单": (REVIEW_PATTERN, "get_sandbox_refund_review", "RFD-<完整核验单号>"),
}


def build_sandbox_tools(store: SandboxStore):
    async def query_order(req, args):
        if not req.authenticated:
            return {"success": False, "error": "沙盒查询需要已验证的账号"}
        order_id = args["order_id"].strip().upper()
        if not re.fullmatch(ORDER_PATTERN, order_id):
            return {"success": False, "error": "请输入完整的沙盒订单号"}
        try:
            ledger = await asyncio.to_thread(store.order_ledger, req.user_id, order_id)
        except sqlite3.Error:
            return {"success": False, "error": "沙盒账单暂时无法读取"}
        if ledger is None:
            return {"success": False, "error": "未找到当前账号的沙盒订单"}
        return {"success": True, "source": "sandbox", **ledger,
                "status_labels": {"settled": "已入账", "authorization_hold": "预授权冻结"},
                "review_command": f"确认提交退款核验 {order_id}"}

    return {
        ORDER_LOOKUP: replace(make_tool(
            ORDER_LOOKUP,
            "核对疑似重复扣款时，若用户提供 ORD-20261024 这类 ORD- 加 8 位数字的订单号，"
            "先用本工具查当前已验证账号下的合成账单，再解释入账和预授权的区别。"
            "如用户需要人工核验，引导用户发送结果中的 review_command。查询本身不创建核验单。",
            {"order_id": {"type": "string", "minLength": 12, "maxLength": 12,
                          "description": "ORD- 加 8 位数字，示例 ORD-20261024；原样使用用户提供的完整编号"}},
            query_order,
            required=["order_id"],
        ), allowed_agents=("billing",)),
    }


async def run_sandbox_command(store: SandboxStore, user_id: str, command: str, argument: str) -> tuple[str, str | None]:
    pattern, tool_name, example = COMMANDS[command]
    identifier = argument.upper()
    if not re.fullmatch(pattern, identifier):
        return f"请输入完整编号，例如「{command} {example}」。", None

    if command == "沙盒查账":
        ledger = await asyncio.to_thread(store.order_ledger, user_id, identifier)
        if ledger is None:
            return "未找到当前账号的沙盒订单。", tool_name
        entries = "\n".join(
            f'{entry["entry_id"]}：{entry["amount_cents"] / 100:.2f} 元，'
            + ("已入账" if entry["status"] == "settled" else "预授权冻结（未入账）")
            for entry in ledger["entries"]
        )
        return (f'沙盒订单 {identifier} 的流水如下\n{entries}\n\n'
                f'如需提交人工核验，请发送「确认提交退款核验 {identifier}」。'), tool_name

    if command == "确认提交退款核验":
        review = await asyncio.to_thread(store.submit_refund_review, user_id, identifier)
        if review is None:
            return "未找到当前账号的沙盒订单。", tool_name
    else:
        review = await asyncio.to_thread(store.refund_review, user_id, identifier)
        if review is None:
            return "未找到当前账号的沙盒退款核验单。", tool_name

    return (f'沙盒退款核验单 {review["review_id"]}\n'
            f'订单 {review["order_id"]}，状态为待人工核验。提交核验不会执行退款。'), tool_name
