import numpy as np

class Maxpool:
    def Forward(self, input):
        h, w, numChannels = input.shape

        # Handle cases where h or w is odd, without zero-padding
        outputH = h // 2
        outputW = w // 2
        if h % 2 != 0:  # Adjust output height for odd h
            outputH += 1
        if w % 2 != 0:  # Adjust output width for odd w
            outputW += 1

        # Initialize output tensor with the correct size
        output = np.zeros((outputH, outputW, numChannels))

        # Perform max pooling
        for channel in range(numChannels):
            for i in range(0, h - 1, 2):
                for j in range(0, w - 1, 2):
                    # Pool over a 2x2 region
                    output[i//2, j//2, channel] = np.max(input[i:i+2, j:j+2, channel])

            if h % 2 != 0:  # Handle last row for odd height
                output[-1, :, channel] = np.max(input[-1, :w:2, channel], axis=0)
            if w % 2 != 0:  # Handle last column for odd width
                output[:, -1, channel] = np.max(input[:h:2, -1, channel], axis=0)

        return output
