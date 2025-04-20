from time import time

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
    prompts_result_path = (
        r"C:\Users\tinte\Root\Repositories\Other\doc2image\prompts_results.txt"
    )

    # Parse the document
    print("[Main] Starting document parsing...")
    parser = PdfParser()
    chunks = Chunkenizer(
        chunk_size=cfg.parser.chunk_size,
        chunk_overlap=cfg.parser.chunk_overlap,
        parser=parser,
    ).split(document_path)
    print(f"[Main] Document parsed into {len(chunks)} chunks.")

    # Start the agent
    print("[Main] Starting agent execution...")
    start = time()
    agent = Agent(
        llm=ChatOllama(model=cfg.agent.model, temperature=cfg.agent.temperature),
        max_chunk_summary_size=cfg.agent.max_chunk_summary_size,
        max_global_summary_size=cfg.agent.max_global_summary_size,
        total_prompts_to_generate=cfg.agent.total_prompts_to_generate,
        summarize_chunk_prompt=cfg.prompts.summarize_chunk.prompt,
        generate_image_prompts_prompt=cfg.prompts.generate_image_prompts.prompt,
        generate_global_summary_prompt=cfg.prompts.generate_global_summary.prompt,
    )
    output: AgentOutput = agent.execute(chunks=chunks)
    end = time()
    print(f"\n[Main] Execution time: {end - start} seconds")

    # Save result in a TXT file
    with open(prompts_result_path, "w") as file:
        for prompt in output.generated_prompts:
            file.write(prompt + "\n\n")

    with open(summary_result_path, "w") as file:
        file.write(output.global_summary)


if __name__ == "__main__":
    main()
