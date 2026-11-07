# Detecting Fake Reviewer Communities on Amazon

**Team:** Federico Trionfetti, Matteo Sorrentini, Alessandro Querqui – MSc in Data Science, Sapienza University of Rome

**Research question:** is it possible to detect suspicious rating behaviour and coordinated groups of reviewers on Amazon?

## Data

Amazon reviews dataset (J. McAuley, J. Leskovec, via The Network Data Repository): 2.9 million reviews from 1996 to 2006, each with user ID, product ID, rating and timestamp.
We focused on **2006**: 517,684 reviews from 260,349 users on 230,332 products.

The data is very sparse (73.7% of users wrote a single review) and strongly skewed towards 5-star ratings (average 4.03).

> The dataset is not included in this repository because of its size. Download it from [The Network Data Repository](https://networkrepository.com/).

## Approach

1. **Network construction:** bipartite users–products graph, projected onto a user–user network. Edges are weighted by the number of co-reviewed products and by rating similarity (Pearson correlation).
2. **Backbone extraction:** kept only statistically significant edges. Final network: 375k nodes, 6.69M edges, scale-free degree distribution (γ ≈ 2.07).
3. **Community detection:** compared Louvain, Leiden and Infomap. Leiden was discarded because of its very low modularity; Louvain and Infomap were combined in a **dual-consensus** step.
4. **Filtering:** suspicious communities = more than 5 members with mean internal correlation above 0.9.
5. **Temporal analysis:**
   - *Burstiness:* checks whether reviews on a product arrive in suspicious bursts.
   - *Baseline deviation:* checks whether the product's rating jumps away from its usual level.

## Key result

Starting from the edges with perfect correlation, we isolated a **clique of 4 users** with 79 co-reviewed products. Of these, 33 products showed bursty review timing and 51 a significant rating deviation; **6 products were flagged by both criteria**, pointing to coordinated rating manipulation.

**Lesson learned:** global community detection had absorbed this small clique into a giant community, diluting its signal. Local edge-level and behavioural analysis was needed to uncover it.

## Files

- `[notebook_name].ipynb` – full analysis (Google Colab)
- `Amazon Review Network Analysis.pdf.pdf` – project presentation

## Tools

Python, NetworkX, igraph, cdlib (Louvain, Leiden, Infomap), pandas, NumPy, SciPy, Matplotlib
