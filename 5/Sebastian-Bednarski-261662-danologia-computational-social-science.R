library(igraph)

# Sebastian Bednarski 261662
# Lista Data science - laboratorium
# Obliczeniowa nauka o sieciach

# Grafy losowe (Erdős-Rényi)
#1. Wygeneruj sieć Erdős-Rényi o stu wierzchołkach i prawdopodobieństwie krawędzi = 0.05.

g <- erdos.renyi.game(n=100, p=0.05)

#2. Wydrukuj podsumowanie grafu - czy graf jest ważony?

summary(g)
plot(g, layout=layout.circle(g))

#Graf nie jest ważony po wygenerowaniu jako sieć Erdős-Rényi, ponieważ nie zostały przypisane wagi do krawędzi.

#3. Wylistuj wszystkie wierzchołki i krawędzie.

E(g)
V(g)
# list wszystkich wierzchołków V, liczba krawedzi E

#4. Ustaw wagi wszystkich krawędzi na losowe z zakresu 0.01 do 1

E(g)$weight <- runif(length(E(g)), min=0.01, max=1)
E(g)
V(g)

#5. Wydrukuj ponownie podsumowanie grafu - czy teraz graf jest ważony?

summary(g)
# Teraz jest ważony!

#6. Jaki jest stopień każdego węzła? Następnie stwórz histogram stopni węzłów.

degree_sequence <- degree(g)
hist(degree_sequence, main = "Histogram Stopni Węzłów", xlab = "Stopień węzła", ylab = "Liczba węzłów")

#7. Ile jest klastrów (connected components) w grafie?

cl <- clusters(g)
cl
num_clusters <- length(cl)
cat("Liczba klastrów w grafie:", num_clusters, "\n")
plot(g, vertex.color=cl$membership)

#8. Zwizualizuj graf w taki sposób, aby rozmiar węzłów odpowiadał mierze PageRank.

pr <- page.rank(g)$vector
plot(g, vertex.size=pr*1000, vertex.label=NA)





#II Grafy preferential attachment (Barabási-Albert)
#1. Wygeneruj graf wedle modelu Barabási-Albert z tysiącem węzłów

g <- barabasi.game(1000)

#2. Zwizualizuj graf layoutem Fruchterman & Reingold

layout <- layout.fruchterman.reingold(g)
plot(g, layout=layout, vertex.size=2,
     vertex.label=NA, edge.arrow.size=.2)

#3. Znajdź najbardziej centralny węzeł według miary betweenness, jaki ma numer?

betweenness(g)
central_node <- which.max(betweenness(g))
cat("Najbardziej centralny węzeł (według betweenness):", central_node, "\n")

#4. Jaka jest średnica grafu?
diameter(g)
cat("Średnica grafu:", diameter(g), "\n")

#5. W komentarzu napisz czym różnią się grafy Barabási-Albert i Erdős-Rényi.

#Graf Barabási-Albert jest modelem preferencyjnego dołączania, co oznacza, że nowe węzły preferują dołączanie do już istniejących węzłów o większym stopniu (popularności). W przypadku grafu Erdős-Rényi, każda para węzłów ma taką samą szansę na utworzenie krawędzi, co prowadzi do losowo połączonych węzłów.
#Graf Barabási-Albert ma rozkład stopni węzłów, który przypomina rozkład potęgowy, podczas gdy graf Erdős-Rényi ma rozkład stopni węzłów zbliżony do rozkładu Poissona.
#Grafy Barabási-Albert są bardziej podatne na powstanie węzłów o bardzo wysokim stopniu (hubów), podczas gdy grafy Erdős-Rényi zwykle nie wykazują takich węzłów o wyjątkowo wysokim stopniu.


#III Rozprzestrzenianie się informacji w sieciach - dane rzeczywiste
#2. Zaimportuj zbiór out.radoslaw_email_email do data.frame i zachowaj tylko pierwsze dwie 
#kolumny (dodatkowo przeskocz dwa pierwsze wiersze), następnie stwórz z tego data 
#frame'a graf skierowany.
dfGraph <- read.csv2("out.radoslaw_email_email", skip=2, sep= " ")[, 1:2]
g <- graph.data.frame(dfGraph, directed = T)

#3. Użyj funkcji simplify aby pozbyć się wielokrotnych krawędzi i pętli. Zweryfikuj czy po tej 
#operacji Twój graf ma 167 węzłów i 5783 krawędzie. Jeśli tak jest, możesz kontynuować.
g <- simplify(g, remove.multiple = T, remove.loops = T)
E(g) #5783/5783 edges from 059b60e (vertex names):
V(g) #167/167 vertices, named, from 059b60e:


#4. Waga na krawędziach niech zostanie ustalona wedle następującego podejścia:
#wij = cntij / cnti, gdzie
#wij - waga na krawędzi pomiędzy węzłęm vi a węzłem vj
#cntij -liczba maili wysłanych przez węzeł vi do węzła vj
#cnti- liczba wszystkich maili wysłanych przez węzeł vi
#Powyższa formuła zakłada, że będziesz musiał(a) użyć pierwotnego data frame’a aby 
#wyliczyć te wagi. Zwróć uwagę, że wedle powyższej formuły suma wag krawędzi 
#wychodzących z każdego węzła będzie wynosiła jeden.

for (edge in E(g)) {
  from_node <- ends(g, edge)[[1]]
  to_node <- ends(g, edge)[[2]]
  cntij <- sum(dfGraph$cnt[dfGraph$from == from_node & dfGraph$to == to_node])
  cnti <- sum(dfGraph$cnt[dfGraph$from == from_node])
  wij <- cntij / cnti
  edge_attr(g, "weight", edge) <- wij
}

#5. Zasymuluj proces rozprzestrzeniania się informacji w grafie wedle następującego opisu:
#• ustaw wszystkim węzłom atrybut activated na FALSE,

V(g)$activated <- FALSE

#• następnie wylosuj lub wybierz (patrz punkt 6) jeden węzeł z grafu i ustaw mu atrybut
#activated na TRUE,
initial_node <- sample(V(g), 1)
V(g)[initial_node]$activated <- TRUE

#• rozpocznij proces rozprzestrzeniania się informacji w grafie za pomocą modelu 
#independent cascades1:
#węzeł, który jest aktywowany, próbuje aktywować swoich sąsiadów z 
#prawdopodobieństwem wij (użyj atrybutu pomocniczego, aby w danej iteracji nie 
#aktywować sąsiadów węzła, który został aktywowany w tej samej iteracji),

#• jeśli węzłowi nie udało się aktywować węzła-sąsiada, nie próbuje aktywować tego 
#węzła w kolejnych interacjach, co nie znaczy, że inny węzeł nie może aktywować 
#tego węzła jako swojego sąsiada,

#• w kolejnych interacjach aktywowane węzły próbują aktywować swoich sąsiadów 
#wedle metody powyżej,

#• gdy niemożliwe jest aktywowanie węzłów, proces się kończy,

#• po każdej iteracji zapisz liczbę aktywowanych węzłów w grafie,

#• powtórz eksperyment stukrotnie dla każdego sposobu wyboru węzła, ponieważ 
#proces jest stochastyczny, następnie uśrednij wyniki dla każdego ze sposobów 
#wyboru węzłą początkowego.
# Funkcja do symulacji rozprzestrzeniania informacji
# Funkcja do symulacji procesu rozprzestrzeniania

#6. Wykonaj powyższy eksperyment dla pięciu różnych węzłów początkowych: (i) węzła o 
#największym outdegree, (ii) dla jednego najbardziej centralnego węzła wedle metody 
#betweenness, (iii) węzła o największym closeness, (iv) dowolnego losowego węzła i (v) 
#węzła wybranego wedle innej miary (napisz jakiej).

nodes_to_test <- c(
  which.max(degree(g, mode="out")),               # (i) największy outdegree
  which.max(betweenness(g)),                      # (ii) największa betweenness
  which.max(closeness(g)),                        # (iii) największa closeness
  sample(V(g), 1),                                # (iv) losowy węzeł
  which.max(degree(g))                            # (v) największy ogólny stopień
)

#7. Jako podsumowanie realizacji zadania przygotuj wykres obrazujący jak przebiegał proces 
#dyfuzji informacji dla różnych węzłów początkowych - na osi X nr kolejnej iteracji, na osi Y
#liczba aktywowanych węzłów w tej iteracji - pięć różnych serii danych - po jednej dla 
#każdego z różnych sprawdzanych węzłów. Do wykresów polecam bibliotekę ggplot2.
library(ggplot2)
data <- data.frame(
  x = rnorm(100), 
  y = rnorm(100)  
)

ggplot(data, aes(x = x, y = y)) + 
  geom_point() + 
  theme_minimal() +
  ggtitle("Wykres rozrzutu losowych danych") +
  xlab("X") +
  ylab("Y")
