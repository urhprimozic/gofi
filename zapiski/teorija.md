# Teorija in fomrulacija problemov
*Work in progress..*
## Iskanje nerazcepnih upodobitev z gradientnimi metodami optimizacije 
Naj bo $G =<S|R>$ končna grupa, $\mathbb F$ polje realnih/kompleksnih števil in $n \in \mathbb N$ dimenzija upodobitve. 
S pomočjo gradientnega spusta iščemo nerazcepne upodobitve $G \to GL_n(\mathbb F)$. 

Naj  za vsak nabor parametrov $\phi \in \mathbb R^P$ preslikava $\hat \rho_\phi \colon G \to GL_n(\mathbb F)$ vsaki matriki priredi matriko in naj bo parametrizacija  $\phi \mapsto \hat \rho_\phi$ gladka. 

Definiramo *funkcije izgub*, ki nam povejo, koliko $\hat \rho$ zmanjka do funkcije, ki slika v unitarne matrike, do homomorfizma in do nerazcepnosti.

Funkcija izgube za relacije (alternativno gremo lahko po vseh $g, h \in G$ in seštevamo $||\hat\rho(gh) - \hat\rho(g)\hat\rho(h)||_F$):
$$
\mathcal L_r = \frac{1}{|R|} \sum \limits_{r \in R} ||\hat \rho (r) - I||_F^2
$$
Funckija izgube za unitarnost (alternativa bi šla samo čez generatorje, kar je že dovolj):
$$
\mathcal L_u = \frac{1}{|G|} \sum \limits_{g \in G} ||\hat\rho(g)\hat\rho(g)^* - I||_F^2
$$
$$
\text{hitrejša alternativa: } \frac{1}{|S|} \sum \limits_{s \in S} ||\hat\rho(s)\hat\rho(s)^* - I||_F^2
$$
Funckija izgube za nerazcepnost:
$$
\mathcal L _{irr} = (||\chi_{\hat \rho}|| - 1)^2 = 
( \left ( \frac{1}{|G|} \sum \limits_{g \in G} tr(\hat \rho (g)) tr(\hat \rho (g^{-1})) \right )  -1)^2
$$
$$
\text{za } \mathbb F = \mathbb C: ( \left ( \frac{1}{|G|} \sum \limits_{g \in G} |tr(\hat \rho (g))|^2 \right )  -1)^2
$$
Definiramo celotno *funkcijo izgube* kot $\mathcal L = \mathcal L_r + \mathcal  L_u + \mathcal L_{irr}$.

**S to funkcijo izgube lahko nerazcepne upodobitve iščemo s pomočjo gradientnega spusta:**
$$
\phi_{n+1} = \phi_n - \eta \nabla\mathcal L(\phi), \quad \phi_0 \in \R^P,
$$
kar je numerični ekvivalent reševanja diferencialne enačbe
$$\frac{d\phi}{dt} = - \nabla \mathcal L (\phi), \quad \phi(0) = \phi_0.
$$
## Particija prostora parametrov
Naj ima grupa $G$ $m$ $n$-dimenzionalnih nerazcepnih upodobitev. Prostor  $\R^P$ začenih parametrov $\phi_0$ modela $\phi \mapsto \hat \rho_\phi$ lahko razbijemo na $m+1$ disjunktnih množic glede na to, k kateri nerazcepni upodobitvi konvergira (divergira) zgornja metoda.

Zanimajo nas lastnosti te particije. 

## Vprašanja
- $\mathcal L$ je polinom. Katere polinome lahko zapišemo kot $\nabla \mathcal L$ neke funkcije izgube?
- povezava med $V(\nabla \mathcal L) = \{\phi \mid \nabla \mathcal L = 0\}$ in mejami med particijami?

## Lastnosti $\mathcal L$
Hitro je videti, da je $\mathcal L_r$ polinom stopnje (največ) $2$, $\mathcal L_u$ in  $\mathcal L_{irr}$ pa polinoma stopnje (največ) $4$.
### Vprašanja in premisleki
- Kateri polinomi so oblike $\mathcal L$ in $\nabla \mathcal L$? Katerin raznoterostim ustreza $\mathcal L = 0$?
- **parametrizacija homomorfizmov:** $\{ f \in fun(G, U_n)\} = V( \mathcal L_r  + \mathcal L_u  )$. Velja namreč $fun(G, U_n) \in fun(G, \mathbb F^{n \times n}) = \mathbb ({F^{n \times n}})^{|G|}$. Homomorfizmi so ravno ničle funkcij izgube, torej raznoterosti. 
- Ali se da isto parametrizacijo narditi tud za $f \colon G \to GL_n$?
- Množice polinomov, ki se pojavijo kot funkcije izgubo za: splošost, za fiksno grupo, za fiksno grupo in fiksno dimenzijo.