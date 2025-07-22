from src.graph import build_graph

if __name__ == "__main__":
    workflow = build_graph()

    state = {
        "user_id":1,
        "messages": [],
        "steps": []
    }

    while True:
        user_input = input("\n👤 Bạn: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Tạm biệt bạn!")
            break

        state["messages"].append({"role": "user", "content": user_input})

        old_len = len(state["messages"])

        for _ in workflow.stream(state):
            new_msgs = state["messages"][old_len:]
            for msg in new_msgs:
                if msg["role"] == "assistant":
                    print(f"\n🤖 Bot: {msg['content']}")
            old_len = len(state["messages"])

        if state.get("result"):
            state["messages"] = []

