# 🖼️ Doc2Image

<p align="center">
  <img src="assets/robot-painting-a-landscape.png?raw=true">
</p>

Doc2Image is an AI-powered app that transforms your documents into creative image ideas. Just upload a file (PDF, TXT, or Markdown) and Doc2Image will read the content, highlight the key points, and create visual descriptions ready to use with your favorite image generation platforms like MidJourney, DALL·E, ChatGPT, and more.

## ✨ Features

- **Beautiful, intuitive interface** — no technical skills required
- **Quick setup** — easy to install and start using
- **Flexible AI support** — works with OpenAI & local models (like LLaMA, Gemma)
- **Prompt history** — keep track of all your generated images
- **Fully customizable** — adjust model settings, summary size, and prompt behavior to fit your workflow

## 📚 How It Works

1. **Upload a document** — PDF, Markdown, or plain text  
2. **Choose a model** — OpenAI or a local Ollama model  
3. **Customize your request** — Select how many image prompts you want, and (optionally) tweak advanced settings like temperature or chunk size  
4. **Generate amazing image ideas** — Doc2Image transforms your document into stunning, ready-to-use prompts

## 🛠️ Getting Started

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
docker run --name doc2image -p 8000:8000 -v data:/app/data doc2image
```
> 💡 This command will create a `data` folder in your current directory to store output files. Do not remove it.

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
