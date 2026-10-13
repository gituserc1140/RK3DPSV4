# RKstudio3Dps: Creative Studio & Ecommerce Toolkit

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rk3dpsv2-d28ge3fupclhhshqdxmaxn.streamlit.app/)

RKstudio3Dps is a Streamlit workspace for taking an idea from AI-assisted concepting through image and 3D creation, content production, and social-ready exports. It supports creators building assets for 3D printing, ecommerce, and social media—not only image-to-3D conversion.

## Features

### Create and build

- Generate images from text prompts with OpenAI
- Generate 3D models with Stability AI
- View GLB models interactively in the browser
- Convert GLB files to OBJ or STL for compatible 3D-printing workflows
- Find 3D-printing resources through quick links to Cults3D and Tinkercad

### Develop ideas with AI

- Create reusable 3D model prompt ideas with Cohere
- Chat with an AI assistant for image prompts, styling ideas, and model concepts
- Apply assistant suggestions directly to image prompts, 3D prompt ideas, blog topics, or social captions
- Draft Markdown blog posts with the Cohere-powered blog writer

### Produce and share content

- Edit videos with trimming, playback-speed controls, fades, black-and-white, mirroring, and audio muting
- Prepare generated images, 3D model previews, and edited videos for download and social posts
- Build social captions with post formats, hashtags, and calls to action
- Open tools for ecommerce and promotion, including Shopify, TikTok Shop, Printful, Etsy, Canva, TikTok, Medium, ChatGPT, and Claude

### App experience

- Progressive web app metadata and service-worker support

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

**Important:** Never commit API keys to version control. Image and 3D model generation keys are entered through password fields in the app for the active browser session and are not read from backend configuration. **Do not commit API keys or a `.env` file.**

For the Cohere AI chat, blog writer, and prompt ideas, users can enter a key in the
app's **Cohere API Key** panel. Alternatively, configure it with an environment
variable or Streamlit secret:

```bash
COHERE_API_KEY=your_cohere_key
```

Start the app:

```bash
streamlit run streamlit_app.py
```

## Troubleshooting OpenAI Image Generation

- If you see an OpenAI error like `billing_hard_limit_reached`, the active key/project has no remaining billing capacity.
- Rotate to a funded key and update `OPENAI_API_KEY` in your local `.env` (or Streamlit secrets).
- The app re-reads `.env` during key lookup, so key rotation is picked up on the next request.

## Streamlit Cloud

Use `streamlit_app.py` as the app entry point. Demo users supply their own OpenAI,
Stability AI, and Cohere credentials in the app; Cohere can also be configured with
`COHERE_API_KEY` in Streamlit secrets or environment variables. The repo is
structured so it can be connected directly to Streamlit Cloud.

MoviePy processing is local to the Streamlit instance and runs only after an edit is
requested. Video editing uses CPU and temporary disk space, so large or long videos
may be slower or exceed the memory/storage limits of Streamlit Community Cloud.

## Attributions 

Attributions to open source technology business providers!
