"""Test script for WebSocket streaming with CameraCapture."""

import asyncio

import cv2

from iris.client.capture.camera import CameraCapture
from iris.client.streaming.websocket_client import StreamingClient


async def display_video(camera: CameraCapture, stop_event: asyncio.Event) -> None:
    """Display video feed in a window.

    Args:
        camera: Camera capture instance
        stop_event: Event to signal when to stop displaying
    """
    while not stop_event.is_set():
        frame = camera.get_frame()

        if frame is not None:
            cv2.imshow("Streaming Feed", frame)

        # Check for 'q' key press or window close
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

        await asyncio.sleep(0.01)


async def main() -> None:
    """Test the WebSocket streaming functionality."""
    camera = None
    client = None
    stop_event = asyncio.Event()

    try:
        # Initialize camera
        camera = CameraCapture(camera_index=0, width=640, height=480, fps=10)

        if not camera.start():
            print("Failed to start camera")
            return

        print("Camera started successfully")

        # Initialize WebSocket client
        server_url = "ws://localhost:8000/ws/stream"
        client = StreamingClient(server_url, camera, jpeg_quality=80)

        print(f"Connecting to {server_url}...")
        print("Streaming... Press 'q' or Ctrl+C to stop")

        # Start streaming and video display tasks
        stream_task = asyncio.create_task(client.stream())
        display_task = asyncio.create_task(display_video(camera, stop_event))

        try:
            # Wait for either task to complete or user interrupt
            done, pending = await asyncio.wait(
                [stream_task, display_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Check if stream_task failed
            for task in done:
                if task == stream_task and task.exception():
                    raise task.exception()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        # Stop streaming and display
        stop_event.set()
        client.stop()

        # Wait for tasks to complete
        await asyncio.gather(stream_task, display_task, return_exceptions=True)

    except ConnectionRefusedError:
        print(f"\nError: Could not connect to server")
        print(
            "Make sure the WebSocket server is running at ws://localhost:8000/ws/stream"
        )
    except OSError as e:
        print(f"\nConnection error: {e}")
        print("The server may not be running or the URL is incorrect")
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up resources
        if client:
            print("\nStatistics:")
            print(f"  Total frames sent: {client.frame_count}")
            print(f"  Average FPS: {client.get_fps():.2f}")

        if camera:
            camera.stop()
            print("Camera stopped")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
