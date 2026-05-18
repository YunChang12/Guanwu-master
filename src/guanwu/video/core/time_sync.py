from __future__ import annotations

import time


class TimeSync:
    def __init__(self) -> None:
        self._start_wall = time.time()
        self._source_video_time = 0.0
        self._sim_time = 0.0

    def tick(self, dt_video: float = 0.1, dt_sim: float = 1.0 / 60.0) -> tuple[float, float, float]:
        self._source_video_time += dt_video
        self._sim_time += dt_sim
        return time.time(), self._sim_time, self._source_video_time

    @property
    def wall_start(self) -> float:
        return self._start_wall
