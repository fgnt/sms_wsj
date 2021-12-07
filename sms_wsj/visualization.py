import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches


def plot_scenario(ex, ax: plt.Axes = None, *, figsize=(12, 8)):
    """
    Plot the source position, sensor position and room dimensions from an
    SMS-WSJ example in the (x, y) plane, i.e. the z-axis is ignored.

    Args:
        ex: An example from SMS-WSJ
        ax:

    Returns:

    """

    if ax is None:
        ax = plt.subplots(1, figsize=figsize)[1]

    speaker_id_to_source_position = {}
    for i, source_position in enumerate(np.array(ex['source_position']).T):
        x, y, z = source_position
        speaker_id = ex["speaker_id"][i]
        if speaker_id in speaker_id_to_source_position:
            np.testing.assert_equal(
                source_position, speaker_id_to_source_position[speaker_id])
        else:
            speaker_id_to_source_position[speaker_id] = source_position
            ax.scatter(x, y, label=f'Speaker {speaker_id}')

    xs, ys, zs = np.array(ex['sensor_position'])
    ax.scatter(xs, ys, label=f'Microphones')

    room_dimensions = ex['room_dimensions']
    (x,), (y,), (z,) = room_dimensions
    # Draw the wall
    w = 0.30
    ax.add_patch(matplotlib.patches.Polygon(np.array([
        (0, 0), (x, 0), (x, y), (0, y), (0, -w), (-w, -w), (-w, y+w),
        (x+w, y+w), (x+w, -w), (0, -w)
    ]), fill=False, hatch='/', linewidth=0))
    ax.add_patch(matplotlib.patches.Rectangle((0, 0), x, y, fill=None))

    ax.set(title=f'Dataset: {ex["dataset"]!r}, ExID: {ex["example_id"]!r}, RT60: {ex["sound_decay_time"]}')
    ax.autoscale(tight=True)
    ax.set_aspect('equal')
    ax.legend()
    return ax
