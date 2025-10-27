from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.distributions import Id, ConstantPermModel
import torch
from gofi.graphs.inversion_table.models import PermDistConnected

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def diagnose_dist(dist) -> None:
    """
    Print quick sanity checks:
      - each model output a_m sums to 1
      - for each (m,h) row, q(..,h,m) sums to 1 over j
      - for each pair i<j, sum_k!=h P(i->k, j->h) == 1  (pair mass)
      - print max/min deficits
    Run this with torch.no_grad() and dist.clear_cache() beforehand.
    """
    with torch.no_grad():
        n = dist.n
        print("=== diagnose_dist ===")
        # 1) model marginals a_m
        print("P(a_m) sums:")
        mv = dist.model_value()
        for m in range(0, n):
            s = float(mv[m-1].sum().cpu().item())
            print(f" m={m}: sum={s:.8f}")

        # 2) q rows
        print("\nq-row sums (for each m,h):")
        max_q_deficit = 0.0
        for m in range(1, n+1):
            for h in range(m, n+1):
                maxpos = n - m + 1
                total = 0.0
                for j in range(1, maxpos+1):
                    total += float(dist.q(j, h, m).cpu().item())
                deficit = abs(total - 1.0)
                max_q_deficit = max(max_q_deficit, deficit)
                if deficit > 1e-6:
                    print(f"  q m={m} h={h} sum={total:.6f} deficit={deficit:.6e}")
        print(" max q deficit:", max_q_deficit)

        # 3) pair mass
        print("\npair mass sums (i<j):")
        max_pair_deficit = 0.0
        for i in range(1, n):
            for j in range(i+1, n+1):
                total = 0.0
                for k in range(1, n+1):
                    for h in range(1, n+1):
                        if k == h: 
                            continue
                        total += float(dist.prob(i, k, j, h).cpu().item())
                deficit = abs(total - 1.0)
                max_pair_deficit = max(max_pair_deficit, deficit)
                if deficit > 1e-6:
                    print(f"  pair (i={i},j={j}) sum={total:.6f} deficit={deficit:.6e}")
        print(" max pair deficit:", max_pair_deficit)

        # 4) spot-check few probabilities
        print("\nspot probs (i=1,j=2,k,h):")
        for k in range(1, min(5, n+1)):
            for h in range(1, min(5, n+1)):
                if k == h: continue
                p = float(dist.prob(1, k, 2, h).cpu().item())
                if p > 0:
                    print(f"  prob(1->{k},2->{h})={p:.6e}")
        print("=====================")


def test_most_probable_permutation_id(n=10):
    print("Testing most_probable_permutation for Identity Model..")
    n = 5
    model = Id(n)
    perm = model.most_probable_permutation()
    assert perm == list(range(1, n + 1)), f"Expected identity permutation, got {perm}"

def test_pair_mass_model(n=5, eps=0.01, layer_size=100):
    print("Testing pair-mass for NN..")

    # Create a constant model where all positions are equally likely
    model = PermDistConnected(n, 4,layer_size, T=10)
    dist = PermModel(model, n)

    # place after building dist in run.py
    with torch.no_grad():
        total = 0.0
        for k in range(1, n+1):
            for h in range(1, n+1):
                if k == h: continue
                total += float(dist.prob(1, k, 2, h))
        
        diff = abs(total - 1.0)
        assert diff < eps, f"Total mass for positions (1,2) is {total}, differs from 1 by {diff}"

def run_diagnosis(n=5, layer_size=100):
    print("Running diagnosis on ConstantPermModel..")
    model = PermDistConnected(n, 4,layer_size, T=10)
    dist = PermModel(model, n)
    with torch.no_grad():
        dist.clear_cache()
        diagnose_dist(dist)

if __name__ == "__main__":
    test_most_probable_permutation_id()
    test_pair_mass_model()
    run_diagnosis()
    print("All tests passed.")