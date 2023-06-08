# DL-SMLFM

## Logs

### Jun 8

We are testing the affection of the filter size `config.filter_size` and filter sigma `config.filter_sigma` to the final result. The loss we use is MSE loss between prediction and label that convolve with a Gaussian kernel with size `7` and siga `(1, 1)`. The branch size is `10`, num of train and validation data is `5000` and `1250`. After 24 epoch trainning and plot the prediction result of each epoch, we find that the predicted pixels tend to place at the boundary of the frame. Same result happen in the first frame generate by `DataLoader(SimDataset(Config(), 1),)` with NumPy random seed set at [0](assets/Logs/Jun-8/0-3-[1,1].tif), [4](assets/Logs/Jun-8/4-3-[1,1].tif), [5](assets/Logs/Jun-8/5-3-[1,1].tif), and [6](assets/Logs/Jun-8/6-3-[1,1].tif). 

Initial guess is that, when calculating the loss, if the prediction is completely wrong, the loss of prediction at boundary always less than others since after convolve a Guassian kernel, half of the Guassian, which is loss if prediction completely wrong, is missing due to the boundary. Thus, the grad point to the label only if the initial prediction is close to the label and has overlap after convolve with Gaussian kernel; otherwise the grad all point to the boundary of frame. 

We modify the [MSE loss](model.py) from

```python
mse_loss = self.mse(self.filter(predi), self.filter(label))
```

to

```python
mse_loss = self.mse(
    self.filter(nn.functional.pad(predi, self.pad_size)), 
    self.filter(nn.functional.pad(label, self.pad_size))
    )
```

where

```python
self.pad_size = (config.filter_size, config.filter_size, 
                 config.filter_size, config.filter_size)
```
