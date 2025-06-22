from autogen import ConversableAgent
import json

class CustomLlamaAgent(ConversableAgent):
    def __init__(self, name, model_fn, system_message="", function_map=None, **kwargs):
        super().__init__(name=name, system_message=system_message, **kwargs)
        self.model_fn = model_fn
        self._function_map = function_map or {}
        if function_map:
            for func_name, func in function_map.items():
                self.register_function({func_name: func})

    def _generate_response(self, messages, **kwargs):
        print(f"[{self.name}] Received messages: {messages}")  # Debug
        prompt = self._format_messages(messages)
        print(f"[{self.name}] Formatted prompt: {prompt}")  # Debug
        last_message = messages[-1] if messages else {}
        content = last_message.get("content", "")
        print(f"[{self.name}] Last message content: {content}")  # Debug

        # Check for function call
        try:
            content_json = json.loads(content) if isinstance(content, str) else content
            if isinstance(content_json, dict) and "function" in content_json:
                func_name = content_json["function"]
                print(f"[{self.name}] Detected function call: {func_name}")  # Debug
                if func_name in self._function_map:
                    func_args = content_json.get("arguments", {})
                    print(f"[{self.name}] Function arguments: {func_args}")  # Debug
                    # Handle function-specific argument requirements
                    if func_name == "get_target_variable_and_metrics":
                        result = self._function_map[func_name](
                            func_args.get("user_prompt", ""),
                            func_args.get("metadata", "")
                        )
                    else:
                        result = self._function_map[func_name](func_args.get("user_prompt", ""))
                    print(f"[{self.name}] Function {func_name} result: {result}")  # Debug

                    return {"content": result.json() if hasattr(result, 'json') else json.dumps(result)}
                else:
                    print(f"[{self.name}] Function {func_name} not found in function_map")  # Debug
                    return {"content": f"Error: Function {func_name} not supported by {self.name}"}
        except json.JSONDecodeError as e:
            print(f"[{self.name}] JSON decode error: {e}")  # Debug

        # Fallback to model_fn
        print(f"[{self.name}] Falling back to model_fn")  # Debug
        response = self.model_fn(prompt)
        print(f"[{self.name}] model_fn response: {response}")  # Debug
        return response

    def _format_messages(self, messages):
        formatted = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        print(f"[{self.name}] Formatted messages: {formatted}")  # Debug
        return formatted