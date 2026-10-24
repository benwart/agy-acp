# ACP Protocol & Antigravity SDK Bridge Design

## Mapping ACP Concepts to `google.antigravity`

| ACP Concept | `google.antigravity` SDK Construct |
|---|---|
| `session/new` (`cwd`, `mcpServers`) | `LocalAgentConfig(workspaces=[cwd], mcp_servers=[McpStdioServer(...), McpStreamableHttpServer(...)], save_dir=persistent_dir)` |
| `session/load` (`sessionId`) | `LocalAgentConfig(conversation_id=sessionId, session_continuation_mode=SessionContinuationMode.RESUME, save_dir=persistent_dir)` |
| `session/set_mode` (`"read-only"`, `"approve"`, `"auto"`) | `CapabilitiesConfig(enabled_tools=BuiltinTools.read_only())` or `policy.ask_user("*", handler=...)` or `policy.allow_all()` |
| `session/set_model` / `configId="model"` | `ModelTarget(name=model_id, endpoint=...)` |
| `configId="reasoning_effort"` | `GeminiModelOptions(thinking_level=ThinkingLevel.LOW \| MEDIUM \| HIGH)` |
| `session/prompt` (`text`, `image`) | `agent.chat([text, from_bytes(data, mime_type=mimeType)])` |
| `Thought` stream chunk | `session/update` -> `{"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": chunk.text}}` |
| `Text` stream chunk | `session/update` -> `{"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": chunk.text}}` |
| `@hooks.pre_tool_call_decide` | Emit `tool_call` (`status="in_progress"`), send `session/request_permission` if gated, return `HookResult(allow=...)` |
| `@hooks.post_tool_call` | Emit `tool_call_update` (`status="completed" \| "failed"`, `rawOutput=...`) |
| `session/cancel` notification | `await response.cancel()` -> resolve `session/prompt` with `{"stopReason": "cancelled"}` |
| `ChatResponse.usage_metadata` | `session/prompt` `result.usage` + `session/update` (`usage_update`) + `_kiro.dev/metadata` |
