import pyperclip

ans = ""    
for n in range(1, 9):
    filename = f"images/dihedral/Dn_1dim_n{n}.pdf"
    label = f"Rešitve gradientnega toka za $n={n}$."
    ans +=f"""
    \\centering
                \\begin{{minipage}}{{0.32\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\linewidth]{{{filename}}}
                    \\caption*{{{label}}}
                \\end{{minipage}} \n
    """
    if n % 3 == 0:
        ans += "\\vspace{0.5em}"
pyperclip.copy(ans)