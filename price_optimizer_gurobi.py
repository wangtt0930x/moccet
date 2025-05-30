from gurobipy import Model, GRB, quicksum

class PriceOptimizer:
    def __init__(self):
        pass

    def optimize_prices(self, tasks, agent_capacity):
        m = Model("test_feasibility")
        m.setParam('OutputFlag', 1)
        m.setParam('TimeLimit', 20)

        accept = {}
        for task in tasks:
            tid = task['id']
            accept[tid] = m.addVar(vtype=GRB.BINARY, name=f"x_{tid}")

        # Maximize number of accepted tasks (no pricing logic)
        m.setObjective(quicksum(accept[task['id']] for task in tasks), GRB.MAXIMIZE)

        m.addConstr(quicksum(task['compute'] * accept[task['id']] for task in tasks) <= agent_capacity)

        m.optimize()

        print(f"Feasibility Test Status: {m.status}")
        results = {}
        if m.status == GRB.OPTIMAL:
            for task in tasks:
                tid = task['id']
                if accept[tid].X > 0.5:
                    results[tid] = "Accepted"
        return results

