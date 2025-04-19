from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseLanguageModel

from doc2image.agent.components import (
    AgentState,
    summarize_chunk_node,
    generate_image_ideas_node,
    continue_summarizing_conditional_edge,
)


@dataclass
class AgentOutput:
    chunks_summaries: list[str]
    generated_ideas: list[str]


class Agent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        max_summary_size: int,
        total_ideas: int,
        summarize_chunk_prompt: str,
        generate_image_ideas_prompt: str,
    ) -> None:
        self.llm = llm
        self.max_summary_size = max_summary_size
        self.total_ideas = total_ideas
        self.summarize_chunk_prompt = summarize_chunk_prompt
        self.generate_image_ideas_prompt = generate_image_ideas_prompt

        self._init_graph()

    def _init_graph(self) -> None:
        # Initialize the state graph
        graph = StateGraph(AgentState)

        # Define the nodes in the graph
        graph.add_node("summarize_chunk", summarize_chunk_node)
        graph.add_node("generate_image_ideas", generate_image_ideas_node)

        # Define the edges in the graph
        graph.add_conditional_edges(
            "summarize_chunk",
            continue_summarizing_conditional_edge,
            {
                "summarize_chunk": "summarize_chunk",
                "generate_image_ideas": "generate_image_ideas",
            },
        )

        # Set the entry point and end nodes
        graph.set_entry_point("summarize_chunk")
        graph.add_edge("generate_image_ideas", END)

        # Compile the graph
        self.graph = graph.compile()

    def execute(self, chunks: list[str]) -> list[str]:
        config = {
            "configurable": {
                "total_chunks": len(chunks),
                "chunks": chunks,
                "summarize_chunk_prompt": self.summarize_chunk_prompt,
                "generate_image_ideas": self.generate_image_ideas_prompt,
                "total_ideas": self.total_ideas,
                "max_summary_size": self.max_summary_size,
                "llm_model": self.llm,
            }
        }
        result = self.graph.invoke(
            {
                "current_chunk_index": 0,
                "chunks_summaries": [],
                "is_sufficient": False,
            },
            config=config,
        )

        return AgentOutput(
            chunks_summaries=result["chunks_summaries"],
            generated_ideas=result["generated_ideas"],
        )
