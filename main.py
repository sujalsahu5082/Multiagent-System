"""
Experiment 8: Multi-Agent System Web UI
Serves an interactive web interface for Planner, Validator, and Worker agents.
"""
from flask import Flask, request, jsonify, render_template_string
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

app = Flask(__name__)

# Define LangGraph State
class AgentState(TypedDict):
    query: str
    plan: str
    validation: str
    valid: bool
    attempts: int
    result: str

llm = ChatOllama(model="llama3.2:1b", temperature=0)

def planner_agent(state: AgentState):
    attempts = state.get("attempts", 0) + 1
    prompt = f"""You are the Planner Agent.
Create a clear, structured, step-by-step plan to accomplish the following task.
Task: {state['query']}

Provide ONLY the plan steps clearly and concisely."""
    response = llm.invoke(prompt).content
    return {"plan": response, "attempts": attempts}

def validator_agent(state: AgentState):
    prompt = f"""You are the Validator Agent.
Evaluate whether the following plan effectively and accurately addresses the task.
Task: {state['query']}
Plan: {state['plan']}

Format your response as:
Status: VALID or INVALID
Reason: <brief explanation>"""
    response = llm.invoke(prompt).content
    is_valid = "STATUS: VALID" in response.upper() or ("VALID" in response.upper() and "INVALID" not in response.upper())
    return {"validation": response, "valid": is_valid}

def worker_agent(state: AgentState):
    prompt = f"""You are the Worker Agent.
Complete the task using the plan below.
Task: {state['query']}
Plan: {state['plan']}

Provide the final answer clearly."""
    response = llm.invoke(prompt).content
    return {"result": response}

def route_after_validation(state: AgentState):
    if state["valid"] or state["attempts"] >= 3:
        return "worker"
    return "planner"

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_agent)
workflow.add_node("validator", validator_agent)
workflow.add_node("worker", worker_agent)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "validator")
workflow.add_conditional_edges("validator", route_after_validation, {"planner": "planner", "worker": "worker"})
workflow.add_edge("worker", END)
multi_agent_app = workflow.compile()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exp-8: Multi-Agent System (LangGraph)</title>
    <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root { --paper: #f4f0e8; --panel: #fffdf8; --ink: #202321; --muted: #68716b; --line: #d8d1c3; --signal: #d94f35; --signal-dark: #ad3927; --teal: #167c78; --teal-soft: #d9eeea; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Space Grotesk', sans-serif; }
        body { background-color: var(--paper); background-image: linear-gradient(rgba(32,35,33,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(32,35,33,0.045) 1px, transparent 1px); background-size: 28px 28px; color: var(--ink); min-height: 100vh; padding: 3.5rem 1rem; }
        .container { max-width: 900px; margin: 0 auto; }
        header { margin-bottom: 2rem; animation: reveal 0.6s ease both; }
        header h1 { font-size: clamp(2.2rem, 6vw, 4.6rem); line-height: 0.95; letter-spacing: 0; max-width: 620px; color: var(--ink); margin-bottom: 0.8rem; }
        header h1::after { content: ''; display: inline-block; width: 0.7em; height: 0.12em; margin-left: 0.15em; vertical-align: 0.12em; background: var(--signal); }
        header p { color: var(--muted); font-size: 1rem; }
        .badge { display: inline-block; padding: 0.3rem 0.65rem; border-radius: 3px; font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 500; margin-top: 0.9rem; background: var(--teal-soft); color: var(--teal); border: 1px solid #a8d7d0; }
        .input-card { background: var(--panel); border: 1px solid var(--line); border-top: 4px solid var(--signal); border-radius: 4px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 8px 8px 0 rgba(32,35,33,0.08); animation: reveal 0.6s 0.1s ease both; }
        .input-group { display: flex; gap: 0.75rem; margin-top: 1rem; }
        input[type="text"] { flex: 1; min-width: 0; padding: 0.85rem 1rem; background: #f8f5ef; border: 1px solid var(--line); border-radius: 3px; color: var(--ink); font-size: 1rem; outline: none; }
        input[type="text"]:focus { border-color: var(--teal); box-shadow: 0 0 0 3px rgba(22,124,120,0.14); }
        button { padding: 0.85rem 1.3rem; background: var(--signal); color: #fffdf8; border: none; border-radius: 3px; font-size: 0.95rem; font-weight: 700; cursor: pointer; transition: background 0.2s, transform 0.2s; }
        button:hover { background: var(--signal-dark); transform: translateY(-2px); }
        button:disabled { background: #aaa69d; cursor: not-allowed; transform: none; }
        .results { display: flex; flex-direction: column; gap: 1rem; }
        .agent-card { background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 1.25rem; animation: reveal 0.5s ease both; }
        .agent-card:nth-child(2) { animation-delay: 0.08s; }
        .agent-card:nth-child(3) { animation-delay: 0.16s; }
        .agent-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; padding-bottom: 0.65rem; border-bottom: 1px solid var(--line); }
        .agent-title { font-weight: 600; font-size: 1.05rem; display: flex; align-items: center; gap: 0.5rem; }
        .agent-content { color: #424943; font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; background: #f8f5ef; border-left: 3px solid var(--teal); padding: 1rem; border-radius: 2px; }
        .tag-valid { background: var(--teal-soft); color: var(--teal); padding: 0.25rem 0.6rem; border-radius: 3px; font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 500; }
        .tag-invalid { background: #f8ddd6; color: var(--signal-dark); padding: 0.25rem 0.6rem; border-radius: 3px; font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 500; }
        .spinner { display: inline-block; width: 1.2rem; height: 1.2rem; border: 2px solid #b7dcd6; border-radius: 50%; border-top-color: var(--teal); animation: spin 1s ease-in-out infinite; margin-left: 0.5rem; vertical-align: middle; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes reveal { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 620px) { body { padding-top: 2rem; } .input-group { flex-direction: column; } button { width: 100%; } }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Multi-Agent System</h1>
            <p>Experiment 8 &bull; LangGraph Autonomous Workflow</p>
            <span class="badge">&#9881; Local Model: llama3.2:1b</span>
        </header>

        <div class="input-card">
            <label for="taskInput" style="font-weight: 500; color: #94a3b8;">Enter Task for the Multi-Agent System:</label>
            <div class="input-group">
                <input type="text" id="taskInput" placeholder="e.g. Explain how AI can help in healthcare" value="Explain how AI can help in healthcare.">
                <button id="runBtn" onclick="runAgents()">Run Agents</button>
            </div>
            <div id="loading" class="hidden" style="margin-top: 1rem; color: #38bdf8; font-size: 0.9rem;">
                <span class="spinner"></span> Agents are collaborating (Planning &rarr; Validating &rarr; Executing)...
            </div>
        </div>

        <div id="resultsArea" class="results hidden">
            <!-- Planner Agent Card -->
            <div class="agent-card">
                <div class="agent-header">
                    <div class="agent-title">&#128221; Planner Agent</div>
                    <span id="attemptBadge" class="badge">Attempt 1</span>
                </div>
                <div id="planContent" class="agent-content"></div>
            </div>

            <!-- Validator Agent Card -->
            <div class="agent-card">
                <div class="agent-header">
                    <div class="agent-title">&#9989; Validator Agent</div>
                    <span id="validBadge" class="tag-valid">VALID</span>
                </div>
                <div id="validContent" class="agent-content"></div>
            </div>

            <!-- Worker Agent Card -->
            <div class="agent-card" style="border-color: #38bdf8;">
                <div class="agent-header">
                    <div class="agent-title" style="color: #38bdf8;">&#128640; Worker Agent (Final Answer)</div>
                </div>
                <div id="workerContent" class="agent-content" style="border: 1px solid #334155;"></div>
            </div>
        </div>
    </div>

    <script>
        async function runAgents() {
            const task = document.getElementById('taskInput').value.trim();
            if (!task) return alert('Please enter a task.');

            const btn = document.getElementById('runBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('resultsArea');

            btn.disabled = true;
            loading.classList.remove('hidden');
            results.classList.add('hidden');

            try {
                const res = await fetch('/api/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task: task })
                });
                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.error || `Request failed (${res.status})`);
                }

                document.getElementById('planContent').textContent = data.plan;
                document.getElementById('attemptBadge').textContent = `Attempts: ${data.attempts}`;
                
                document.getElementById('validContent').textContent = data.validation;
                const badge = document.getElementById('validBadge');
                if (data.valid) {
                    badge.className = 'tag-valid';
                    badge.textContent = 'VALID';
                } else {
                    badge.className = 'tag-invalid';
                    badge.textContent = 'INVALID (Retried)';
                }

                document.getElementById('workerContent').textContent = data.result;
                results.classList.remove('hidden');
            } catch (err) {
                alert('Error running agents: ' + err.message);
            } finally {
                btn.disabled = false;
                loading.classList.add('hidden');
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/run", methods=["POST"])
def run_agents():
    data = request.get_json() or {}
    task = data.get("task", "Explain how AI can help in healthcare.")
    
    initial_state = {
        "query": task,
        "plan": "",
        "validation": "",
        "valid": False,
        "attempts": 0,
        "result": ""
    }
    
    try:
        final_state = multi_agent_app.invoke(initial_state)
    except Exception as error:
        app.logger.exception("Agent run failed")
        return jsonify({"error": str(error)}), 500
    
    return jsonify({
        "plan": final_state["plan"],
        "validation": final_state["validation"],
        "valid": final_state["valid"],
        "attempts": final_state["attempts"],
        "result": final_state["result"]
    })

if __name__ == "__main__":
    print("Starting Multi-Agent Web App on http://127.0.0.1:5005 ...")
    app.run(host="0.0.0.0", port=5005)