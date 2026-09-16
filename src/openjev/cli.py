import argparse
import json

from .domain import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description="DecisionTics: typed decisions, real Doom")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the local browser cockpit")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1", help="Bind address; containers use 0.0.0.0")
    sub.add_parser("setup", help="Download the pinned model for OPENJEV_BACKEND (mlx or cpu)")
    decide = sub.add_parser("decide", help="Score context/questions/candidates from a JSON file")
    decide.add_argument("input", help="Path to a DecisionRequest JSON file")
    train = sub.add_parser("train", help="Reproduce the local policy with the train extra")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--samples", type=int, default=60000)
    evaluate = sub.add_parser("evaluate", help="Matched-seed local/rules/random comparison")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=1000)
    evaluate.add_argument("--scenario", choices=SCENARIOS, default=SCENARIOS[0])
    evaluate.add_argument("--output", default="runs/evaluation.json")
    play = sub.add_parser("play", help="Play one complete headless episode")
    play.add_argument("--policy", choices=("local", "rules", "random", "jev", "language"), default="local")
    play.add_argument("--instruction", help="English instruction for the language controller")
    play.add_argument("--scenario", choices=SCENARIOS, default=SCENARIOS[0])
    play.add_argument("--seed", type=int, default=42)
    play.add_argument("--directive", choices=("hunt", "conserve", "pacifist"), default="hunt")
    play.add_argument("--record", help="Output path prefix for GIF, trace and episode metrics")
    args = parser.parse_args()
    if args.command == "setup":
        from .decisions import download_model

        print(download_model())
    elif args.command == "decide":
        from pathlib import Path

        from .decisions import DecisionRequest, DecisionService

        request = DecisionRequest.model_validate_json(Path(args.input).read_text())
        service = DecisionService()
        try:
            print(service.decide(request).model_dump_json(indent=2))
        finally:
            service.close()
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("openjev.server:app", host=args.host, port=args.port, access_log=False)
    elif args.command == "train":
        if args.samples <= 0 or args.epochs <= 0:
            parser.error("samples and epochs must be positive")
        from .train import train

        train(samples=args.samples, epochs=args.epochs)
    elif args.command == "evaluate":
        if not 1 <= args.episodes <= 1000:
            parser.error("episodes must be between 1 and 1000")
        from .evaluate import evaluate

        evaluate(args.output, episodes=args.episodes, seed=args.seed, scenario=args.scenario)
    else:
        from .evaluate import run_episode

        print(
            json.dumps(
                run_episode(args.policy, args.scenario, args.seed, args.directive, args.record, args.instruction),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
