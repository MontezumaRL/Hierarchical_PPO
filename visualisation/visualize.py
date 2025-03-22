import matplotlib.pyplot as plt
import numpy as np

def visualize_state(observation, color=False):
    """
    Visualize the current state (observation) from the environment.
    Parameters:
    observation: np.array
        The observation (state) returned by the environment.
    color: boolean
        If you want the observation be shown as an rgb image or grayscale.
    """
    if color:
        plt.imshow(observation)
    else:
        plt.imshow(observation, cmap="gray")
        
    plt.axis('off')
    plt.show()


def visualize_frame_stack(frame_stacker):
    """
    Visualizes the frames in the frame stack using matplotlib.
    
    Parameters:
    frame_stacker: FrameStacker
        The FrameStacker object containing the stack of frames.
    """

    frames = list(frame_stacker.frames)
    num_frames = len(frames)

    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

    if num_frames == 1:
        axes = [axes]

    for i, frame in enumerate(frames):
        axes[i].imshow(frame, cmap='gray')  # Display the frame as a grayscale image
        axes[i].axis('off')  # Hide the axis

    plt.show()
