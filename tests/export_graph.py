from pathlib import Path
import json
from invoice_processor.agent import invoice_agent

output = Path("artifacts") / "invoice_graph.json"
output.parent.mkdir(parents=True, exist_ok=True)

graph = invoice_agent.get_graph()
output.write_text(json.dumps(graph.to_json(), indent=2))
print(f"Grafo guardado en {output}")
