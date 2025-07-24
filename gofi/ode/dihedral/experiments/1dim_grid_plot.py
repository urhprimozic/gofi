import matplotlib.pyplot as plt
from gofi.plot.colors import cmap_blue, cmap_orange
import numpy as np
import argparse
import pickle

#cmaps  =["OrRd", "YlGn", cmap_orange, cmap_blue]
cmaps  =["OrRd", "YlGn", "Oranges", "Blues"]

def closest_corner(x, y):
    # Define the 4 possible corners
    candidates = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Compute squared distance to each
    closest = min(candidates, key=lambda p: (x - p[0])**2 + (y - p[1])**2)
    closest_index = candidates.index(closest)
    return closest_index
def smooth_time(time, joint=30):
    if time < joint:
        return time 
    return joint

def plot_grid(min_value, max_value, resolution, grid_dict : dict, filename : str, n : int, time_len=False, t_max=1):
    """
    If time_len = True, time of convergence will be measured by the nubmer of steps instead of time till losss  < eps
    
    """


    grids = [np.zeros((resolution, resolution)) -1 for i in range(4) ] 
    # coordinate grid
    values = np.linspace(min_value, max_value, resolution)
    points_r, points_s = np.meshgrid(*[values] * 2)
    iterator = zip(points_r.flatten(), points_s.flatten())

    max_times = [0, 0, 0, 0]
    for index, (r, s) in enumerate(iterator):
        # collect solution data
        solution = grid_dict[(r.item(), s.item())]
        # catch errors at computation
        if solution is None:
            # save to "closest grid"
            limit = closest_corner(r,s)
            row = index % resolution
            column = index // resolution
            grids[limit][row][column] = t_max
            max_times[limit] = max(max_times[limit], time)
            continue

        if time_len:
            time=smooth_time(len(solution.t))

        else:
            # extract time
            if solution.t_events is None:
                time = t_max
            elif solution.t_events[0].shape == (0,):
                time = t_max
            else:
                #print(solution.t_events, " ", solution.t_events[0].shape, "  !!!!!\n" )
                time = solution.t_events[0][0]
        #time = solution.t[-1]
        pr = solution.y[0]
        ps = solution.y[1]
        limit = closest_corner(pr[-1], ps[-1])
        # update max time
        max_times[limit] = max(max_times[limit], time)

        # save to the right grid
        row = index % resolution
        column = index // resolution
        grids[limit][row][column] = time

    # plot grids
    #ims = []
    #caxs = []
    #candidates = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(4):
        grid = grids[i].astype(np.float32)
        cmap = cmaps[i]



        #norm = Normalize(vmin=0, vmax=max_times[i])
        #plt.imshow(grid, cmap=cmap,norm=norm, alpha=(grid != -1).astype(np.float32))
        # plot just this grid
        plt.imshow(grid, cmap=cmap,vmin=0, vmax=max_times[i], alpha=(grid != -1).astype(np.float32))
       # plt.colorbar()
        #ims.append(im)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #caxs.append(cax)
        #fig.colorbar(im, cax=cax, label=f"Converges to ${candidates[i]}$")

   
    plt.title(f"Limit points and convergence speed of initial parameters\n$\\hat \\rho \\colon D_{{2\\cdot{n}}} \\to R$")
    plt.xlabel("r")
    plt.ylabel("s")
    ticks_res = 6
    labels = np.linspace(min_value, max_value, ticks_res)
    labels = [round(x, 2) for x in labels ]
    plt.xticks(np.linspace(0, resolution, ticks_res) , labels )  # positions, labels
    plt.yticks(np.linspace(0, resolution, ticks_res) , labels )  # positions, labels
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=str, help="Order of rotation . Encodes n at D2n.")
    parser.add_argument("min_value", type=str, help="Minimal value of the grid")
    parser.add_argument("max_value", type=str, help="Max value of the grid")
    parser.add_argument("resolution", type=str, help="Grid resolution")
    parser.add_argument("filename", type=str, help="Filename of the grid")
    parser.add_argument("plotname", type=str, help="Filename of the plot to save")
    parser.add_argument("--t_max", default="1",type=str, help="Maximum time of integration")
    args = parser.parse_args()
    # collect args
    n = int(args.n)
    min_value = int(args.min_value)
    max_value = int(args.max_value)
    resolution = int(args.resolution)
    filename = args.filename
    plotname = args.plotname
    t_max = float(args.t_max)

    # load grid
    with open(f'{filename}', 'rb') as f:
        grid_dict = pickle.load(f)
    plot_grid(min_value, max_value, resolution, grid_dict, plotname, n, t_max=t_max)