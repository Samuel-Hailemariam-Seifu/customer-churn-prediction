import argparse

import uvicorn

from src.data.eda import run_eda
from src.models.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Customer churn prediction project entrypoint")
    parser.add_argument(
        "--task",
        choices=["eda", "train", "serve"],
        required=True,
        help="Task to run",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.task == "eda":
        run_eda()
    elif args.task == "train":
        run_training()
    else:
        uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
