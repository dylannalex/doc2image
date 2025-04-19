import hydra

from langchain_ollama.chat_models import ChatOllama

from doc2image.agent.core import Agent, AgentOutput
from doc2image.docs import PdfParser, Chunkenizer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg) -> None:
    document_path = (
        r"C:\Users\tinte\Root\Repositories\Other\doc2image\data\Prompt Engineering.pdf"
    )
    summary_result_path = (
        r"C:\Users\tinte\Root\Repositories\Other\doc2image\summary_result.txt"
    )
    ideas_result_path = (
        r"C:\Users\tinte\Root\Repositories\Other\doc2image\ideas_result.txt"
    )

    parser = PdfParser()
    chunks = Chunkenizer(
        chunk_size=cfg.parser.chunk_size,
        chunk_overlap=cfg.parser.chunk_overlap,
        parser=parser,
    ).split(document_path)

    agent = Agent(
        llm=ChatOllama(model=cfg.agent.model, temperature=cfg.agent.temperature),
        max_summary_size=cfg.agent.max_summary_size,
        total_ideas=cfg.agent.total_ideas,
        summarize_chunk_prompt=cfg.prompts.summarize_chunk.prompt,
        generate_image_ideas_prompt=cfg.prompts.generate_image_ideas.prompt,
    )
    output: AgentOutput = agent.execute(chunks=chunks)

    # Save result in a TXT file
    with open(ideas_result_path, "w") as file:
        for idea in output.generated_ideas:
            file.write(idea + "\n\n")

    with open(summary_result_path, "w") as file:
        for summary in output.chunks_summaries:
            file.write(summary + "\n\n")

if __name__ == "__main__":
    main()
