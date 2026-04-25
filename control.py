
"""
Control Chrome dinosaur game using blinks. Adjust THRESHOLD to adjust sensitivity.
"""

################################################################################
# IMPORTS
################################################################################

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pynput.keyboard import Key, Controller

################################################################################
# CONSTANTS
################################################################################

KEY = Key.space
TARGET_CHANNEL_INDEX = 0                # index [0, 1, 2, 3] = channels [1, 2, 3, 4]
THRESHOLD = 15                          # adjust for sensitivity

################################################################################
# MAIN
################################################################################

def main():
    """
    Set up the connection to the Ganglion board and begin keystroke detection loop. 
    """

    params = BrainFlowInputParams()
    board_id = BoardIds.GANGLION_NATIVE_BOARD
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    # ensure we have channels available
    if not eeg_channels:
        print("No EEG channels found.")
        return

    target_channel = eeg_channels[TARGET_CHANNEL_INDEX]
    print(f"Streaming from channel: {target_channel}.")

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    keyboard = Controller()
    state = False

    # rolling buffer for plotting
    buffer_len = sampling_rate * 3
    plot_buffer = deque(iterable=[0.0]*buffer_len, maxlen=buffer_len)

    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot(np.arange(buffer_len), np.zeros(buffer_len), lw=1)

    def update(frame):
        nonlocal state
        new_data = board.get_board_data()

        if new_data.shape[1] > 0:

            # extract data from target channel
            channel_data = new_data[target_channel]
            plot_buffer.extend(channel_data)

            window_size = 40
            if len(plot_buffer) >= window_size:
                raw_signal = np.array(list(plot_buffer)[-window_size:])
                signal_energy = np.std(raw_signal)
                if signal_energy > THRESHOLD:
                    if not state:
                        keyboard.tap(KEY)
                        print(f"Jumped! Energy was {signal_energy}.")
                        state = True
                    else:
                        state = False

        # refresh plot
        y_data = np.array(plot_buffer)
        y_data = y_data - np.mean(y_data) # reject DC
        line.set_ydata(y_data)
        ax.set_ylim(-100, 100)

        return (line,)

    try:
        ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
        plt.title(f"EEG Signal (Channel {TARGET_CHANNEL_INDEX+1})")
        plt.show()
    finally:
        board.stop_stream()
        board.release_session()
        print("Session Ended.")


################################################################################
# ENTRY POINT
################################################################################

if __name__ == "__main__":
    main()
