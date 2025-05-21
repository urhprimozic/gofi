

if __name__ == "__main__":
    from gofi.ode.dihedral import Dihedral
    from gofi.ode.dihedral.calculus import DihedralCalculus

    # Example usage
    dihedral = Dihedral()
    calculus = DihedralCalculus(dihedral)
    result = calculus.some_method()
    print(result)