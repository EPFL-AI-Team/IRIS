"""Test script for CameraCapture class."""

import time

import cv2

from iris.client.capture.camera import CameraCapture


def main() -> None:
    """Test the camera capture functionality."""
    # Initialize camera
    camera = CameraCapture(camera_index=1, width=640, height=480, fps=30)

    if not camera.start():
        print("Failed to start camera")
        return

    print("Camera started. Press 'q' to quit.")
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            frame = camera.get_frame()

            if frame is not None:
                cv2.imshow("Camera Test", frame)
                frame_count += 1

            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Calculate and display statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print("\nStatistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Elapsed time: {elapsed:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")

        # Clean up
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
