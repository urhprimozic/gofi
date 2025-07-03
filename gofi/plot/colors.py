from matplotlib.colors import LinearSegmentedColormap

# Define normalized RGB colors
black  = (0.0, 0.0, 0.0)
darkblue   =  (8 / 255, 80 / 255, 160 / 255)
blue   = (8 / 255, 96 / 255, 204 / 255)
lightblue   = (20 / 255, 110 / 255, 227 / 255)
lightorange = (240/ 255, 199/ 255, 164/ 255)
alphaorange = (243 /255, 111 / 255, 33 / 255, 0.5)
orange = (243 / 255, 111 / 255, 33 / 255)
darkorange = (200 / 255, 91 / 255, 9 / 255)
green =(0, 1, 159 / 255)
yellow = (1.0, 1.0, 0.5)
white  = (1.0, 1.0, 1.0)

cmap_blue = LinearSegmentedColormap.from_list(
    "blue",
[darkblue, blue,  white],
    N=600  # More steps = smoother gradient
)
cmap_orange = LinearSegmentedColormap.from_list(
    "orange",
[darkorange, orange,  white],
    N=600  # More steps = smoother gradient
)

# Create smooth colormap from black → blue → orange → yellow → white
blueorange = LinearSegmentedColormap.from_list(
    "blueorange",
    [black, blue, lightorange, white],
    N=512  # More steps = smoother gradient
)
cmap_blue_orange_blackless = LinearSegmentedColormap.from_list(
    "blueorange_blackless",
[blue, orange, lightorange, white],
    N=600  # More steps = smoother gradient
)

cmap_dakblue_blue = LinearSegmentedColormap.from_list(
    "darkblue_blue",
    [darkblue, blue, lightblue],
    N=600  # More steps = smoother gradient
    )