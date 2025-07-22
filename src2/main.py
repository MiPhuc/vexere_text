from src2.graph import build_graph

if __name__ == "__main__":
    workflow = build_graph()

    state = {
        "messages": [],
        "steps": []
    }

    print("📢 Xin chào! Tôi có thể giúp gì cho bạn về đặt vé, huỷ vé, đổi giờ, hoặc tra cứu?")

    while True:
        user_input = input("\n👤 Bạn: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Tạm biệt bạn!")
            break

        state["messages"].append({"role": "user", "content": user_input})

        for _ in workflow.stream(state):
            pass

        last_msg = [m for m in state["messages"] if m["role"] == "assistant"][-1]
        print(f"\n🤖 Bot: {last_msg['content']}")

        if state.get("result"):
            state["messages"] = []  # reset cho vòng tiếp theo
