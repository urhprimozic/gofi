import argparse
from gofi.groups import get_S_n
import textwrap



if __name__ == '__main__':  
    parser = argparse.ArgumentParser( epilog=textwrap.dedent('''\
         Trains a model of S_n --> on a grid of initial parameters.
         '''))
    parser.add_argument("n", type=int, help="Degree (n) of S_n.")
    parser.add_argument("min_param", type=float, help="Minimal value for parameters")
    parser.add_argument("max_param", type=float, help="Degree (n) of S_n.")
    parser.add_argument("min_x", type=int, help="Degree (n) of S_n.")
    parser.add_argument("min_x", type=int, help="Degree (n) of S_n.")
    args = parser.parse_args()