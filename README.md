# 🖼️ Doc2Image

<p align="center">
  <img src="assets/robot-painting-a-landscape.png?raw=true">
</p>

Turn any document into a gallery of AI‑ready image ideas. Upload any file (PDF, DOCX, TXT, and more) and Doc2Image will read it, summarize the content, and generate a list of unique visual concepts you can take to the image generator of your choice. Perfect for blog posts, presentations, decks, social posts—or just sparking your imagination.

## Why You’ll Love It

- **Intuitive Interface:** A clean, guided experience from start to finish.
- **Flexible AI:** Use OpenAI models or go local via Ollama—your call.
- **Idea History:** Never lose a great idea with the built-in idea gallery.
- **Customizable:** Fine-tune the AI's creativity to get the perfect results.
- **Budget-friendly:** works great with small models (e.g., `gpt-4.1-nano`, `deepseek-r1:1.5b`), so **it’s really cheap to run**.

## How It Works (3 Quick Steps)

1. **Configure your models (one‑time):** Add API keys and pick your providers in Settings.
2. **Generate amazing image ideas:** Upload a document and let the app craft tailored visual prompts.
3. **Browse your Idea Gallery:** Revisit past sessions and reuse your favorite prompts.

> **Disclaimer:** Doc2Image does not generate images. It generates image ideas (prompts) you can paste into any AI image generator (e.g., Grok, ChatGPT, WhatsApp, etc.).

##  Demo

https://github.com/user-attachments/assets/ed499dfd-7326-4788-a419-fdb4852e55a9


## Getting Started

You can run doc2image in two ways depending on your needs:

- **Basic setup** — the simplest, uses only OpenAI models

- **Advanced setup** — supports both OpenAI and local models via Ollama

> 💡 **Pre-requisite:** Make sure you have [Docker](https://docs.docker.com/get-started/get-docker/) installed on your system.

### Basic Setup

1. Open your terminal or command line.
2. Pull the latest image from Docker Hub:

```bash
docker pull dylantinten/doc2image:latest
```

3. Run the application:

```bash
docker run --name doc2image -p 8000:8000 -v data:/app/data dylantinten/doc2image:latest
```
> 💡 This command will create a `data` folder in your current working directory to store output files (do not delete this folder). You can change `data` to any path you prefer, or `cd` into the directory where you want your data to live before running the command.

4. Open your browser and visit: [http://localhost:8000](http://localhost:8000). You're ready to go!

To stop the application:

```bash
docker stop doc2image
```

To start it again:

```bash
docker start doc2image
```

### Advanced Setup

This setup runs both the doc2image app and an Ollama server locally using Docker Compose. You’ll be able to generate prompts using both OpenAI and open-source models like LLaMA or Gemma.

1. Download the `docker-compose.yaml`.

2. Open your terminal or command line and navigate where `docker-compose.yaml` is located.

> 💡 This command will create a `data` folder in your current working directory to store output files (do not delete this folder).

3. Build and launch the services:

```bash
docker compose up --build
```

4. Open your browser and go to: [http://localhost:8000](http://localhost:8000) to start using the app.

To stop the services:

```bash
docker compose down
```

To restart them later:

```bash
docker compose up
```

## ❤️ Contributing

We’d love your help to make Doc2Image even better!  

Whether it’s reporting bugs, suggesting new features, or submitting a pull request — all contributions are welcome.

If you enjoy using this project, **please consider giving it a star ⭐️** — it helps others discover it too!
