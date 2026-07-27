from langchain.agents import create_agent


# sk-94fe8fbb7176469cb95380f41b589d5e
def get_weather(city: str) -> str:
    """_summary_

    Args:
        city (str): _description_

    Returns:
        str: _description_
    """
    return f"{city}天气"


agent = create_agent(model="deepseek-chat", tools=[get_weather], system_prompt="你是一个天气预报员")

agent.invoke({"messages": [{"role": "user", "content": "查询北京天气"}]})
