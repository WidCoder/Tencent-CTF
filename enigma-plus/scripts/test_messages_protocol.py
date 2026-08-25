import argparse
import time
import requests

TOOLS = [{
    "name": "Bash",
    "description": "Execute a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

def run_case(name, url, model, payload, timeout):
    start = time.perf_counter()
    try:
        r = requests.post(
            url,
            headers={
                "x-api-key": "EMPTY",
                "Authorization": "Bearer EMPTY",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=(5, timeout),
        )
        elapsed = time.perf_counter() - start
        print(f"{name:28} status={r.status_code} time={elapsed:.2f}s body={r.text[:300]}")
    except requests.Timeout as e:
        elapsed = time.perf_counter() - start
        print(f"{name:28} TIMEOUT time={elapsed:.2f}s error={e}")
    except Exception as e:
        print(f"{name:28} ERROR {type(e).__name__}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:23106/v1/messages")
    parser.add_argument("--model", default="glm52_10")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    base = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "top_p": 0.95,
        "system": "Reply briefly.",
    }

    run_case(
        "1_baseline",
        args.url,
        args.model,
        {**base, "messages": [{"role": "user", "content": "Reply OK."}]},
        args.timeout,
    )

    run_case(
        "2_tools_only",
        args.url,
        args.model,
        {
            **base,
            "tools": TOOLS,
            "messages": [{"role": "user", "content": "Reply OK. Do not call a tool."}],
        },
        args.timeout,
    )

    assistant_tool = {
        "type": "tool_use",
        "id": "call-test",
        "name": "Bash",
        "input": {"command": "echo test"},
    }

    run_case(
        "3_tool_result_only",
        args.url,
        args.model,
        {
            **base,
            "tools": TOOLS,
            "messages": [
                {"role": "user", "content": "Run the command."},
                {"role": "assistant", "content": [assistant_tool]},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call-test",
                        "content": "test",
                    }],
                },
            ],
        },
        args.timeout,
    )

    run_case(
        "4_tool_result_plus_text",
        args.url,
        args.model,
        {
            **base,
            "tools": TOOLS,
            "messages": [
                {"role": "user", "content": "Run the command."},
                {"role": "assistant", "content": [assistant_tool]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-test",
                            "content": "test",
                        },
                        {
                            "type": "text",
                            "text": "Continue with the next step.",
                        },
                    ],
                },
            ],
        },
        args.timeout,
    )

if __name__ == "__main__":
    main()