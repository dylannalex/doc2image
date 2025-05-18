import os
from time import time

import hydra

from .prompt import Prompt
from .pipeline import DocumentSummarizer, ImagePromptsGenerator
from .docs import chunkenize_document
from .llm import create_llm


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg) -> None:
    document_path = input("Enter the document path: ")
    save_path = input(r'Enter the save path (default: ".\output\"): ') or os.path.join(
        os.getcwd(), "output"
    )
    summary_result_path = os.path.join(save_path, "summary_result.txt")
    prompts_result_path = os.path.join(save_path, "prompts_results.txt")
    llm_model = input("Enter the LLM model (default: 'gemma3:1b'): ") or "gemma3:1b"

    # Parse the document
    print("[Main] Starting document parsing...")

    chunks = chunkenize_document(
        document_path,
        chunk_size=cfg.parser.chunk_size,
        chunk_overlap=cfg.parser.chunk_overlap,
        separators=cfg.parser.separators,
        is_separator_regex=cfg.parser.is_separator_regex,
        keep_separator=cfg.parser.keep_separator,
        strip_whitespace=cfg.parser.strip_whitespace,
    )

    print(f"[Main] Document parsed into {len(chunks)} chunks.")

    # Start the doc summarization execution
    print("[Main] Starting doc summarization execution...")

    start = time()
    doc_summerizer = DocumentSummarizer(
        llm=create_llm(
            api_name="ollama",
            model_name=llm_model,
            temperature=cfg.pipeline.document_summarizer.llm_params.temperature,
            top_p=cfg.pipeline.document_summarizer.llm_params.top_p,
            top_k=cfg.pipeline.document_summarizer.llm_params.top_k,
        ),
        document_chunks=chunks,
        max_document_summary_size=cfg.pipeline.document_summarizer.max_document_summary_size,
        max_chunk_summary_size=cfg.pipeline.document_summarizer.max_chunk_summary_size,
        summarize_chunk_prompt=Prompt(
            messages=cfg.prompts.summarize_chunk.messages,
            parameters=cfg.prompts.summarize_chunk.parameters,
        ),
        generate_document_summary_prompt=Prompt(
            messages=cfg.prompts.generate_document_summary.messages,
            parameters=cfg.prompts.generate_document_summary.parameters,
        ),
    )
    document_summary, chunk_summaries = doc_summerizer.run()
    end = time()
    print(f"[Main] Document summary:\n```\n{document_summary}\n```")
    print(f"[Main] Doc summarization execution time: {end - start} seconds")

    # Start the image prompts generation execution
    print("[Main] Starting image prompts generation execution...")
    start = time()
    image_prompts_generator = ImagePromptsGenerator(
        llm=create_llm(
            api_name="ollama",
            model_name=llm_model,
            temperature=cfg.pipeline.image_prompts_generator.llm_params.temperature,
            top_p=cfg.pipeline.image_prompts_generator.llm_params.top_p,
            top_k=cfg.pipeline.image_prompts_generator.llm_params.top_k,
        ),
        document_summary=document_summary,
        total_prompts_to_generate=cfg.pipeline.image_prompts_generator.total_prompts_to_generate,
        generate_image_prompts_prompt=Prompt(
            messages=cfg.prompts.generate_image_prompts.messages,
            parameters=cfg.prompts.generate_image_prompts.parameters,
        ),
    )
    image_prompts = image_prompts_generator.run()
    end = time()
    print(f"[Main] Image prompts:\n```\n{image_prompts}\n```")
    print(f"[Main] Image prompts generation execution time: {end - start} seconds")

    # Save result in a TXT file
    os.makedirs(save_path, exist_ok=True)

    with open(prompts_result_path, "w") as file:
        for prompt in image_prompts:
            file.write(prompt + "\n\n")

    with open(summary_result_path, "w") as file:
        file.write(document_summary)
        file.write("\n\n" + "-" * 20 + "\n\n")
        for chunk_summary in chunk_summaries:
            file.write(chunk_summary + "\n\n")


if __name__ == "__main__":
    main()
