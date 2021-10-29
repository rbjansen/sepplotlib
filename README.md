# sepplotlib

> Separation plots for binary classification problems.

## Credits
> The one-dimensional separation plot is adapted from code originally produced by [Brian Greenhill, Michael D. Ward, and Audrey Sacks](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-5907.2011.00525.x). 
The bi-separation plot and model criticism plot are adapted from code originally produced by [Michael Colaresi and Zuhaib Mahmood](https://journals.sagepub.com/doi/10.1177/0022343316682065).

## Installation

`pip install sepplotlib` to install.


## Example usage

Please see the accompanied notebook for an example using mock data.

All functions expect a pandas DataFrame, and strings for the relevant columns. To generate a one-dimensional separation plot for instance, simply run:

```python
SeparationPlot(
	df=df, 
	y_true="y_true", 
	y_pred="y_pred", 
	title="Example"
)
```

Similarly to generate a model criticism plot:

```python
ModelCriticismPlot(
	df=df, 
	y_true="y_true", 
	y_pred="y_pred", 
	lab="lab", 
	title="Example"
)
```

And finally, to generate a two-dimensional, bi-separation plot:

```python
BiseparationPlot(
    df=df,
    x="y_pred_a",
    y="y_pred_b",
    obs="y_true",
    lab="lab",
    title="Example",
)
```
