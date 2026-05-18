from __future__ import annotations


def planner_mock_next_action(store, goal: str | None = None) -> dict:
    raise NotImplementedError("Default dummy mock planner is disabled. Please integrate a real planning API.")
