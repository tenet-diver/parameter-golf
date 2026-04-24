import unittest

from fastest.scripts.run_cheap_screen_candidate import (
    run_next_cheap_screen_candidate,
    select_next_candidate,
)


class Fast12NextCheapScreenCandidateTest(unittest.TestCase):
    def test_select_next_candidate_prefers_lowest_status_order_then_created_at(self) -> None:
        backlog = [
            {
                "taskId": "FAST-20",
                "status": "queued",
                "statusOrder": 5,
                "createdAt": "2026-04-24T00:00:10Z",
                "lane": "cheap-screen",
                "candidateId": "exp-b",
            },
            {
                "taskId": "FAST-19",
                "status": "queued",
                "statusOrder": 5,
                "createdAt": "2026-04-24T00:00:05Z",
                "lane": "cheap-screen",
                "candidateId": "exp-a",
            },
            {
                "taskId": "FAST-18",
                "status": "in_progress",
                "statusOrder": 1,
                "createdAt": "2026-04-24T00:00:01Z",
                "lane": "cheap-screen",
                "candidateId": "exp-in-progress",
            },
            {
                "taskId": "FAST-17",
                "status": "queued",
                "statusOrder": 0,
                "createdAt": "2026-04-24T00:00:01Z",
                "lane": "ablation",
                "candidateId": "exp-ablation",
            },
        ]

        selected = select_next_candidate(backlog, lane="cheap-screen")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["taskId"], "FAST-19")
        self.assertEqual(selected["candidateId"], "exp-a")

    def test_run_next_candidate_returns_noop_when_no_cheap_screen_task_available(self) -> None:
        backlog = [
            {
                "taskId": "FAST-22",
                "status": "queued",
                "statusOrder": 0,
                "createdAt": "2026-04-24T00:00:00Z",
                "lane": "ablation",
                "candidateId": "exp-ablation",
            }
        ]
        runner_calls = 0

        def runner(_: dict) -> dict:
            nonlocal runner_calls
            runner_calls += 1
            return {"status": "success"}

        outcome = run_next_cheap_screen_candidate(backlog, runner=runner)
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(outcome["reasonCode"], "no-task-available")
        self.assertEqual(runner_calls, 0)


if __name__ == "__main__":
    unittest.main()
