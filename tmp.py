plots = [
    "plot_C10_n=10_m=7.png",
    "plot_C15_n=15_m=12.png",
    "plot_C7_n=7_m=5.png",
    "plot_C12_n=12_m=11.png",
    "plot_C16_n=16_m=16.png",
    "plot_C7_n=7_m=7.png",
    "plot_C12_n=12_m=7.png",
    "plot_C17_n=17_m=12.png",
    "plot_C8_n=8_m=5.png",
    "plot_C12_n=12_m=8.png",
    "plot_C18_n=18_m=15.png",
    "plot_C9_n=9_m=6.png",
    "plot_C13_n=13_m=13.png",
    "plot_C19_n=19_m=17.png",
    "plot_C14_n=14_m=12.png",
    "plot_C5_n=5_m=5.png",
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
                \\includegraphics[
                  width=0.49\\textwidth
                ] {{images/actions/{p}}}
        """
        if begin:
            text = (
                f"""%
            \\begin {{figure*}}[ht]
                \\centering
                """
                + text
            )
            begin = False
        else:
            text = text + "\\end {{figure*}}\n"
            begin = True

        f.write(text)
        # f.write("\n")

        text_original = f"""%
                    \\begin {{figure*}}[ht]
                \\centering
                \\includegraphics[
                  width=0.75\\textwidth
                ] {{images/actions/{p}}}
                \\caption {{Stohastične matrike 
                $P_s$ na začetku in na koncu 
                iskanja delovanja ciklične grupe $C_{{{n}}}$ na
                množici $[{m}]$. 
                }}
              \\end {{figure*}}
        %"""
