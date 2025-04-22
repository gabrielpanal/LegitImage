import numpy as np

class Conv:
    def __init__(self, inputChannels, numFilters, kernel=None):
        self.numFilters = numFilters
        # Initialize filters with shape (3, 3, inputChannels, numFilters)
        self.filters = kernel if kernel is not None else np.random.randn(3, 3, inputChannels, numFilters) / 9

    def ReLU(self, output):
        return np.maximum(0, output)

    def GenerateRegions(self, image, h, w):
        for height in range(h - 2):
            for width in range(w - 2):
                # Extract a 3x3 patch across all input channels
                imageRegion = image[height:height+3, width:width+3, :]
                yield imageRegion, height, width

    def Forward(self, input):
        if len(input.shape) == 2:
            # If the input is single-channel, reshape to (h, w, 1)
            input = np.expand_dims(input, axis=-1)

        self.recentInput = input
        h, w, numChannels = input.shape  # Now numChannels represents the input channels

        # Prepare the output tensor (height, width, numFilters)
        outputH = h - (self.filters.shape[0] - 1)  # 3x3 filter -> subtract 2 from height
        outputW = w - (self.filters.shape[1] - 1)  # 3x3 filter -> subtract 2 from width
        output = np.zeros((outputH, outputW, self.numFilters))
        
        # Perform convolution across each filter
        for filterIndex in range(self.numFilters):
            for imageRegion, height, width in self.GenerateRegions(input, h, w):
                # Apply the filter to the image region across all input channels
                # The filters are now in the format (3, 3, inputChannels, numFilters)
                # So we need to sum across the input channels
                output[height, width, filterIndex] = np.sum(
                    imageRegion * self.filters[:, :, :, filterIndex]
                )

        return self.ReLU(output)

    def Backward(self, gradLossOut, learnRate):
        gradLossFilters = np.zeros(self.filters.shape)
        gradLossIn = np.zeros(self.recentInput.shape)
        h, w = self.recentInput.shape[:2]

        for imageRegion, height, width in self.GenerateRegions(self.recentInput, h, w):
            for outFilter in range(self.numFilters):
                # Accumulate gradient for the filters
                gradLossFilters[:, :, :, outFilter] += gradLossOut[height, width, outFilter] * imageRegion
                # Backpropagate the gradient to the input (also across all input channels)
                gradLossIn[height:height+3, width:width+3, :] += gradLossOut[height, width, outFilter] * self.filters[:, :, :, outFilter]

        # Update filters
        self.filters -= learnRate * gradLossFilters
        return gradLossIn
