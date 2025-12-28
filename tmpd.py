plots = [
    "plot_D10_n=10_m=8.png",
    "plot_D19_n=19_m=19.png",
    "plot_D8_n=8_m=8.png",
    "plot_D18_n=18_m=15.png",
    "plot_D6_n=6_m=5.png",
    "plot_D9_n=9_m=9.png",
]

# sort by n, then by m
plots = sorted(
    plots, key=lambda x: (int(x.split("_")[1][1:]), int(x.split("_")[3][2:-4]))
)

if len(plots) % 2 != 0:
    plots = plots[:-1]

with open("tmp.txt", "w") as f:
    begin = True
    for p in plots:
        splits = p.split("_")
        n = splits[1][1:]
        m = splits[3][2:-4]

        text = f"""%
            \\begin{{figure}}[ht]
              \\centering
              \\includegraphics[
                width=0.75\\textwidth
              ]{{images/actions/{p}}}
            \\end{{figure}}
        %"""
        f.write(text)
        # f.write("\n")
