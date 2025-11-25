import argparse
from uuid import uuid4
from langchain_core.messages import SystemMessage, HumanMessage
from .agent import invoice_agent
from .prompts import INVOICE_PROMPT
from .slack_handler import run_slack_bot


def console_main():
    thread_id = f"console-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        text = input(">> ")
        state = invoice_agent.get_state(config)
        is_new = state is None or not state.values.get("messages")

        messages = (
            [SystemMessage(content=INVOICE_PROMPT), HumanMessage(content=text)]
            if is_new else [HumanMessage(content=text)]
        )

        result = invoice_agent.invoke({"messages": messages}, config=config)
        if hasattr(result, "structured_response"):
            print(result.structured_response.summary)
        else:
            print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["console", "slack"], default="console")
    args = parser.parse_args()

    if args.mode == "slack":
        run_slack_bot()
    else:
        console_main()


if __name__ == "__main__":
    main()
