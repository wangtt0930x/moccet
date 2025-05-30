from gurobipy import Model, GRB, quicksum

class PriceOptimizer:
    def __init__(self, min_margin=0.05, risk_weight=0.0):
        self.min_margin = min_margin
        self.risk_weight = risk_weight

    def optimize_prices(self, tasks, agent_capacity):
        m = Model("dynamic_pricing")
        m.setParam('OutputFlag', 0)

        price = {}
        accept = {}

        for task in tasks:
            tid = task['id']
            price[tid] = m.addVar(lb=task['base_cost'] * (1 + self.min_margin), name=f"p_{tid}")
            accept[tid] = m.addVar(vtype=GRB.BINARY, name=f"x_{tid}")

        m.setObjective(
            quicksum(accept[tid] * (price[tid] - task['base_cost']) for task in tasks for tid in [task['id']]) -
            self.risk_weight * quicksum((1 - task['confidence']) * accept[task['id']] for task in tasks),
            GRB.MAXIMIZE
        )

        m.addConstr(
            quicksum(task['compute'] * accept[task['id']] for task in tasks) <= agent_capacity,
            "compute_limit"
        )

        # Removed hard urgency constraint to avoid infeasibility

        m.optimize()

        print(f"Optimization Status: {m.status}")
        results = {}

        if m.status == GRB.OPTIMAL:
            print("✅ Optimized tasks:")
            for task in tasks:
                tid = task['id']
                if accept[tid].X > 0.5:
                    results[tid] = price[tid].X
                    print(f"Task {tid}: ${price[tid].X:.2f}")
        else:
            print("⚠️ No optimal solution found. Status code:", m.status)

        return results

