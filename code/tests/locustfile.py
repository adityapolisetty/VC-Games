"""
Locust load testing for VC Card Game

Usage:
  # Web UI mode (recommended)
  locust -f locustfile.py --host=http://localhost:8765

  # Headless mode
  locust -f locustfile.py --host=http://localhost:8765 --users 10 --spawn-rate 2 --run-time 60s --headless

  # Test Railway deployment
  locust -f locustfile.py --host=https://your-app.railway.app --users 20 --spawn-rate 5 --run-time 120s --headless
"""

from locust import HttpUser, task, between, events
import random
import time

class VCCardGameUser(HttpUser):
    """
    Simulates a user playing the VC Card Game
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.team_name = None
        self.stage = 0

    def on_start(self):
        """Called when a user starts - generate unique team name"""
        self.team_name = f"LoadTest_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        self.stage = 0

    @task(10)
    def play_complete_game(self):
        """
        Complete game flow - highest weight (most common scenario)
        """
        try:
            # Stage 0: Start game
            signal_type = random.choice(['median', 'top2'])
            response = self.client.get("/", name="GET Landing Page")

            if response.status_code != 200:
                return

            # Submit team name and signal type
            response = self.client.post("/", data={
                "team_name": self.team_name,
                "signal_type": signal_type
            }, name="POST Start Game")

            if response.status_code != 200:
                return

            # Stage 1: Purchase signals and make investments
            num_signals = random.randint(0, 5)  # 0-5 signals
            num_investments = random.randint(1, 9)  # 1-9 piles

            stage1_data = {}

            # Purchase signals
            for i in range(num_signals):
                pile_idx = random.randint(0, 8)
                stage1_data[f"signal_{pile_idx}"] = "on"

            # Make investments (total should be <= 100)
            budget = 100
            investments = self._generate_random_investments(budget, num_investments)

            for pile_idx, amount in investments.items():
                stage1_data[f"invest_{pile_idx}"] = str(amount)

            response = self.client.post("/stage1", data=stage1_data, name="POST Stage 1")

            if response.status_code != 200:
                return

            # Stage 2: Additional investments (only in Stage 1 piles)
            stage2_data = {}
            remaining_budget = budget - sum(investments.values())

            if remaining_budget > 0:
                stage2_investments = self._generate_random_investments(
                    remaining_budget,
                    min(len(investments), 3)
                )

                for pile_idx, amount in stage2_investments.items():
                    if pile_idx in investments:  # Only invest in Stage 1 piles
                        stage2_data[f"invest2_{pile_idx}"] = str(amount)

            response = self.client.post("/stage2", data=stage2_data, name="POST Stage 2")

            if response.status_code != 200:
                return

            # Stage 3: View results
            response = self.client.get("/results", name="GET Results")

        except Exception as e:
            print(f"Error in play_complete_game: {e}")

    @task(3)
    def restart_from_stage1(self):
        """
        Start game and immediately restart - tests restart functionality
        """
        try:
            # Start game
            self.client.get("/", name="GET Landing (Restart)")

            response = self.client.post("/", data={
                "team_name": f"{self.team_name}_restart",
                "signal_type": "median"
            }, name="POST Start (Restart)")

            if response.status_code != 200:
                return

            # Immediately restart
            response = self.client.get("/restart", name="GET Restart")

        except Exception as e:
            print(f"Error in restart_from_stage1: {e}")

    @task(1)
    def view_leaderboard_only(self):
        """
        Complete game and view results multiple times - tests leaderboard loading
        """
        try:
            # Quick game
            self.client.get("/", name="GET Landing (Leaderboard)")

            self.client.post("/", data={
                "team_name": f"{self.team_name}_leaderboard",
                "signal_type": "median"
            }, name="POST Start (Leaderboard)")

            # Minimal Stage 1
            self.client.post("/stage1", data={
                "invest_0": "100"
            }, name="POST Stage 1 (Leaderboard)")

            # Skip Stage 2
            self.client.post("/stage2", data={}, name="POST Stage 2 (Leaderboard)")

            # View results - this loads leaderboard
            self.client.get("/results", name="GET Results (Leaderboard)")

        except Exception as e:
            print(f"Error in view_leaderboard_only: {e}")

    def _generate_random_investments(self, budget, num_piles):
        """
        Generate random investments that sum to <= budget
        """
        if num_piles == 0 or budget <= 0:
            return {}

        investments = {}
        remaining_budget = budget
        piles = random.sample(range(9), min(num_piles, 9))

        for i, pile_idx in enumerate(piles):
            if i == len(piles) - 1:
                # Last investment gets remaining budget
                amount = remaining_budget
            else:
                # Random amount up to remaining budget
                max_amount = remaining_budget // (len(piles) - i)
                amount = random.randint(1, max(1, max_amount))

            if amount > 0:
                investments[pile_idx] = amount
                remaining_budget -= amount

        return investments


class HighConcurrencyUser(HttpUser):
    """
    Simulates high concurrency user - rapid requests
    """
    wait_time = between(0.1, 0.5)  # Very short wait time

    @task
    def rapid_landing_page_requests(self):
        """Rapid landing page access"""
        self.client.get("/", name="GET Landing (High Concurrency)")


class StressTestUser(HttpUser):
    """
    Stress test user - pushes boundaries
    """
    wait_time = between(0.5, 1)

    @task
    def max_signals_max_piles(self):
        """Test maximum signal purchases and investments"""
        try:
            team_name = f"Stress_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

            self.client.get("/", name="GET Landing (Stress)")

            self.client.post("/", data={
                "team_name": team_name,
                "signal_type": "median"
            }, name="POST Start (Stress)")

            # Purchase all signals
            stage1_data = {}
            for i in range(9):
                stage1_data[f"signal_{i}"] = "on"

            # Invest in all piles
            investment_per_pile = 100 // 9
            for i in range(9):
                stage1_data[f"invest_{i}"] = str(investment_per_pile)

            self.client.post("/stage1", data=stage1_data, name="POST Stage 1 (Stress)")

            self.client.post("/stage2", data={}, name="POST Stage 2 (Stress)")

            self.client.get("/results", name="GET Results (Stress)")

        except Exception as e:
            print(f"Error in stress test: {e}")


# Event listeners for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test finished.")


# Custom stats if needed
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Custom request handler - can add custom metrics here
    """
    if exception:
        print(f"Request failed: {name} - {exception}")

    # Log slow requests
    if response_time > 2000:  # More than 2 seconds
        print(f"SLOW REQUEST: {name} took {response_time}ms")
