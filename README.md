# TODO
## upodobitve
- zrihtaj kodo za upodobitve Dn
- ko najdeš eno upodobitev u, dodaj v loss pogoj, da je karakter pravokoten na u
- narediš loss kot produkt lossov za različne dimenzije matrik  --> imaš nekaj, kar išče po vseh dimenzijah 
## grafi 
- dodaj avtomorfizme med dvema grafoma
- a, b iz A_n naključno --> cayley graf
- x y druga dva --> cayley graf   
- poglej, če je izomorfizem 
- probaš na vseh cayleyevih grafih A_n glede na dva naključno izbrana elementa 

# Plani za naprej:
- dodaj loss ortogonalnosti na prejšnje upodobitve
- dodaj 
- Ogromne simulacije na calculusu
- Pripravi berljive zvezke


gemini Pro vs chatgbt


# Irreducable representations gradient search
### Implementacija ideje Urbana Jezernika 

**[Povezava do teorije: ](zapiski/teorija.md)**

We have a model 
$$\phi  \mapsto \hat \rho _\phi : G \to \mathbb{F}^{\text{dim} \times \text{dim}}$$
of an unknown irreducable representation $ \rho _\phi : G \to GL_{\text{dim}}\mathbb{F}$.

Using gradient optimisation, we want to change the parameters $\phi \in \mathbb F ^p$ of the model to minimize 
$(1 - ||\chi_{\hat \rho_\phi}||)^2$ and TODO. 

We study, how initial parameters $\phi_0$ effect the convergence/divergence of the optimisation. 
In the future, we might focus on the gradient flow 
$$
\frac{d\phi}{dt} = - \nabla \mathbf{L} (\phi)
$$

## Results
### $\rho \colon S_3 \to \mathbb \R \setminus \{0\}$
The following resoults are found in [src/analysis.ipynb](src/analysis.ipynb).


Example of model $\hat \rho \colon S_3 \to \mathbb R$ converging to $\text{sign} \colon S_3 \to \mathbb R \setminus \{0\}$:
![model for S3 -> R converges to sign](demo/S3_converged_to_sign.png)
### Different parameters of $\hat \rho_{\phi=(p_1, p_2)} \colon S_3 \to \mathbb R$
Model $\hat \rho $ of $\rho \colon S_3 \to \mathbb R \setminus \{0\}$ is given by 
$$
\hat \rho((1,2)) = p_1 \quad \hat \rho((1,3))=p_2.
$$
Optimising $p_1$ and $p_2$ with adam, learning rate of $0.01$ and $p_1, p_2 \in (-\frac{1}{2},\frac{1}{2})$ yields the following result:
![$S_3 \to \mathbb R$](demo/S3->R_100x100_-0.5-0.5.png)
Convergence means that the loss was lower than $\varepsilon = 0.001$. Divergence means that the run didn't converge in 400 steps. 
If loss is small, $\phi=(p_1, p_2) \approx (1,1)$ (converges to $\text{id}$) or $\phi=(p_1, p_2) \approx (-1,-1)$ (converges to $\text{sign}$).

### Irreducability search for $\rho \colon S_3 \to GL_2 \mathbb R$ and $\rho \colon S_3 \to GL_2 \mathbb C$


Some results:

![Konvergenca](demo/converged.png)
![Lokalni ekstrem, ki ni globalni.](demo/failed_to_converge.png)
![Blizu konvergenci - verjetno gre za napačno izbiro hiperparametrov.](demo/nearly_converge.png)

# TODO
- izpelji na roko za 1dim, 2dim in lepe G= S_n, D_n, C_n
- testiraj Sn na R \to 2x2





