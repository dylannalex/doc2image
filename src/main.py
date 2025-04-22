from time import time

import hydra

from doc2image.agent.core import Agent, AgentOutput
from doc2image.docs import chunkenize_document
from doc2image.llm import load_llm_model


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

    llm = load_llm_model(
        api_name=cfg.agent.llm.api,
        model_name=cfg.agent.llm.model_name,
        temperature=cfg.agent.llm.temperature,
        top_p=cfg.agent.llm.top_p,
        top_k=cfg.agent.llm.top_k,
    )

    # Parse the document
    print("[Main] Starting document parsing...")

    chunks = chunkenize_document(
        document_path, cfg.parser.chunk_size, cfg.parser.chunk_overlap
    )

    print(f"[Main] Document parsed into {len(chunks)} chunks.")

    # Start the agent
    print("[Main] Starting agent execution...")
    start = time()
    agent = Agent(
        llm=llm,
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
