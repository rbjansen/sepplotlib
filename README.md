# sepplotlib

> Separation plots for binary classification problems.

## Credits
> The one-dimensional separation plot is adapted from code originally produced by [Brian Greenhill, Michael D. Ward, and Audrey Sacks](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-5907.2011.00525.x). 
The bi-separation plot and model criticism plot are adapted from code originally produced by [Michael Colaresi and Zuhaib Mahmood](https://journals.sagepub.com/doi/10.1177/0022343316682065).

## Installation

`pip install sepplotlib` to install.


## Example usage

Please see the accompanied notebook for an example using mock data.

The included figures are objects that expect a pandas DataFrame and strings for the relevant columns. To generate a one-dimensional separation plot for instance, simply run:

```python
import sepplotlib as spl
spl.SeparationPlot(
    df=df,
    y_true="y_true",
    y_pred="y_pred",
    title="Example"
)
```

<img src="https://user-images.githubusercontent.com/31345940/139453276-2caf6b1c-087f-40a9-baa2-2c3fc8f79ab2.png" width="500">

Similarly to generate a model criticism plot:

```python
import sepplotlib as spl
spl.ModelCriticismPlot(
    df=df,
    y_true="y_true",
    y_pred="y_pred",
    lab="lab",
    title="Example"
)
```

<img src="https://user-images.githubusercontent.com/31345940/139453840-e9469065-8a67-42d7-81fc-61dac823df32.png" width="400">

And finally, to generate a two-dimensional, bi-separation plot:

```python
import sepplotlib as spl
spl.BiseparationPlot(
    df=df,
    x="y_pred_a",
    y="y_pred_b",
    obs="y_true",
    lab="lab",
    title="Example",
)
```

<img src="https://user-images.githubusercontent.com/31345940/139453518-83a4ad72-ffba-442c-816c-35902fcaf5b1.png" width="400">

Please run `help` on any of these classes to learn what can be customized (e.g. `help(spl.SeparationPlot)`).
