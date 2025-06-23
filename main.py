import argparse
import uvicorn
from app import app

def main():
    parser = argparse.ArgumentParser(description="Lite RAG Web Service")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port to run the server on"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()