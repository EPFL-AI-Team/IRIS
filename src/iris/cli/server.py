import sys


def run():
    """Run the server with proper signal handling."""
    try:
        from iris.server.app import main
        main()
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        print("\n\nServer startup interrupted. Exiting gracefully...", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    run()
