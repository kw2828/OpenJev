# Hosting OpenJev

There are two separate deployments:

- **GitHub Pages browser playground:** [kw2828.github.io/OpenJev](https://kw2828.github.io/OpenJev/). WebLLM runs the language model on the visitor's GPU, alongside recorded gameplay. The Python API and live Doom engine are not hosted by Pages. See [browser details](browser-model.md).
- **Interactive app:** the Docker image runs the decision API, CPU language model, real headless ViZDoom, and browser interface together. Use a Docker host or Hugging Face Docker Space. No external inference service or paid API key is required.

## Run the full app in Docker

```sh
git clone https://github.com/kw2828/OpenJev.git
cd OpenJev
docker build -t openjev .
docker run --rm --name openjev -p 127.0.0.1:7860:7860 --memory=6g --cpus=2 openjev
```

Open http://localhost:7860. The first build downloads pinned public model weights; scoring uses the image's cache offline. Start with the tiny imitation baseline for responsive gameplay, then select the language controller to try English instructions. CPU language scoring advances the simulation more slowly than wall time.

Use a machine with at least 6 GB available RAM and sufficient disk for the image/model. The image runs as UID 1000 on port 7860. `/healthz` checks server availability, not model readiness; `/api/model` reports lazy loading and errors. Start **one server worker and one replica**: the Doom arena and model queue live in memory.

The container deliberately uses **Qwen/Qwen3-0.6B, float32, CPU**, pinned to `c1899de289a04d12100db370d81485cdf75e47ca`, with thinking disabled. This is a smaller, different checkpoint from the Mac MLX Qwen3-4B model. Its accuracy and gameplay may differ; the existing 4B results and GIF are not evidence for 0.6B. Responses expose model, revision and backend. The same bounded candidate scoring contract applies. No generated explanation or reasoning trace is used.

## Validation in this update

The image was built and exercised on Linux ARM64 through Docker Desktop: real CPU scoring and three language-controlled Doom steps. See `evidence/hosting-smoke.json`. The test used a 6 GB container memory limit and two CPUs. The x86-64 hosting build and an actual remote Space/Codespace have not yet been exercised; resource needs and latency should be checked on the chosen host. Mac tests and a container smoke run are not active GitHub CI.

## Hugging Face Docker Space

1. Create a Space in your own Hugging Face account, choose **Docker**, and select CPU hardware. Current Hugging Face policy requires a paid plan to create Docker Spaces even though CPU Basic has no hourly charge. This is not the free-hosting path; use GitHub Pages for the browser playground. Verify [current policy](https://huggingface.co/docs/hub/spaces-overview) before provisioning.
2. Upload this repo's `Dockerfile`, `pyproject.toml`, `LICENSE`, and `src/` directory. Copy `deploy/space-README.md` to the Space's root as `README.md`. That file supplies `sdk: docker` and port 7860.
3. Let the image build. `SPACE_HOST` supplies the public HTTPS origin automatically. If the host does not expose it, set `OPENJEV_PUBLIC_ORIGIN=https://YOUR-SPACE.hf.space` in the Space variables.
4. Open the app. Confirm `/api/model` shows the CPU checkpoint, score a request, then start Doom. A cold start or sleeping Space may take time.

GitHub updates do not automatically sync a separate Space. Upload the new revision when updating it. No Hugging Face Space has been provisioned by this repository update; it needs an authenticated owner account.

See the [official Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker).

## Other Docker hosts

Route HTTPS to port 7860 and set the exact public origin:

```sh
docker run --rm -p 127.0.0.1:7860:7860 \
  -e OPENJEV_PUBLIC_ORIGIN=https://demo.example.org openjev
```

Your reverse proxy must preserve the Host header. Unknown hosts and cross-origin browser requests are rejected. The app does not trust arbitrary forwarded-host headers or enable CORS. Keep the default single process; use your platform's HTTPS proxy and access controls.

`OPENJEV_PUBLIC_DEMO=1` is the container default. It disables the paid TypeSafe API even if a key is accidentally present, and displays a shared-demo notice. **Everyone using one server shares the same Doom controls, state, and English instruction.** Do not enter private text. Decision playground requests are not added to game history or stored by the app; hosting providers may have their own logs. This is a small shared research demo, not a multi-tenant service. The model accepts one running and one queued request, returning 429 beyond that; this is a capacity bound, not authentication or per-user rate limiting. Put a login or rate limit at the host if needed.

## GitHub Codespaces

[Open a Codespace](https://codespaces.new/kw2828/OpenJev) to build the supplied `.devcontainer` and start the full app on forwarded port 7860. Choose at least 8 GB RAM. The start script configures the Codespaces HTTPS origin; the forwarded port stays private by default. Stop/delete the Codespace when finished. This is a developer launch path, not an always-on public deployment. Codespaces billing and availability depend on your account.

## GitHub Pages

Pages serves `docs/index.html` from `main:/docs` with `.nojekyll`. The WebLLM worker runs Qwen3-0.6B on the visitor's GPU; separate recorded 4B and tiny-baseline runs retain explicit labels. Pages does not execute the Python API or ViZDoom engine; see [GitHub's Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages). No backend API keys or model weights are published into the static site.
