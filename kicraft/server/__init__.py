"""KiCraft server: the capped agent-loop backend (Track B).

Everything that can spend money flows through this package's capped client
(`client.CappedOpenRouterClient`) and spend gate (`spend_guard.SpendGuard`), so
the cost-safety guarantees from the plan's "B0. Cost-safety architecture" are
enforced in one place and cannot be bypassed by a new code path.
"""
