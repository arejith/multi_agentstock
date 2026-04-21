def get_tool_sets(news_retriever, fundamentals_retriever, transformer_retriever, writer, keyword_llm=None):
    from tools.analysis_tools.risk_tool import create_risk_tool
    from tools.analysis_tools.sentiment_tool import create_sentiment_tool
    from tools.analysis_tools.transformer_tool import create_transformer_tool
    from tools.memory_tools.read_tools.fundamentals_tool import create_fundamentals_tool
    from tools.memory_tools.read_tools.news_tool import create_news_tool
    from tools.memory_tools.read_tools.transformer_tool import create_transformer_input_tool
    from tools.memory_tools.write_tools.fundamentals_write_tool import (
        create_fundamentals_write_tool,
    )
    from tools.memory_tools.write_tools.news_write_tool import create_news_write_tool
    from tools.memory_tools.write_tools.transformer_write_tool import (
        create_transformer_write_tool,
    )

    data_tools = [
        create_news_write_tool(writer, llm=keyword_llm),
        create_fundamentals_write_tool(writer),
        create_transformer_write_tool(writer),
    ]

    analysis_tools = [
        create_news_tool(news_retriever),
        create_fundamentals_tool(fundamentals_retriever),
        create_transformer_input_tool(transformer_retriever),
        create_sentiment_tool(news_retriever),
        create_risk_tool(fundamentals_retriever),
        create_transformer_tool(),
    ]

    return {
        "data_tools": data_tools,
        "analysis_tools": analysis_tools,
        "all_tools": data_tools + analysis_tools,
    }
