from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain.chat_models.base import BaseChatModel


from doc2image.agent.components import (
    AgentState,
    summarize_chunk_node,
    generate_global_summary_node,
    generate_image_prompts_node,
    continue_summarizing_conditional_edge,
)


@dataclass
class AgentOutput:
    global_summary: str
    generated_prompts: list[str]


class Agent:
    def __init__(
        self,
        llm: BaseChatModel,
        max_chunk_summary_size: int,
        max_global_summary_size: int,
        total_prompts_to_generate: int,
        summarize_chunk_prompt: str,
        generate_image_prompts_prompt: str,
        generate_global_summary_prompt: str,
    ) -> None:
        self.llm = llm
        self.max_chunk_summary_size = max_chunk_summary_size
        self.max_global_summary_size = max_global_summary_size
        self.total_prompts_to_generate = total_prompts_to_generate
        self.summarize_chunk_prompt = summarize_chunk_prompt
        self.generate_image_prompts_prompt = generate_image_prompts_prompt
        self.generate_global_summary_prompt = generate_global_summary_prompt
        self._init_graph()

    def _init_graph(self) -> None:
        # Initialize the state graph
        graph = StateGraph(AgentState)

        # Define the nodes in the graph
        graph.add_node("summarize_chunk", summarize_chunk_node)
        graph.add_node("generate_global_summary", generate_global_summary_node)
        graph.add_node("generate_image_prompts", generate_image_prompts_node)

        # Define the edges in the graph
        graph.add_conditional_edges(
            "summarize_chunk",
            continue_summarizing_conditional_edge,
            {
                "summarize_chunk": "summarize_chunk",
                "generate_global_summary": "generate_global_summary",
            },
        )
        graph.add_edge("generate_global_summary", "generate_image_prompts")

        # Set the entry point and end nodes
        graph.set_entry_point("summarize_chunk")
        graph.add_edge("generate_image_prompts", END)

        # Compile the graph
        self.graph = graph.compile()

    def execute(self, chunks: list[str]) -> list[str]:
        config = {
            "configurable": {
                "total_chunks": len(chunks),
                "chunks": chunks,
                "summarize_chunk_prompt": self.summarize_chunk_prompt,
                "generate_image_prompts_prompt": self.generate_image_prompts_prompt,
                "generate_global_summary_prompt": self.generate_global_summary_prompt,
                "total_prompts_to_generate": self.total_prompts_to_generate,
                "max_chunk_summary_size": self.max_chunk_summary_size,
                "max_global_summary_size": self.max_global_summary_size,
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
            global_summary=result["global_summary"],
            generated_prompts=result["image_prompts"],
        )
